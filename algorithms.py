import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from dinov3.models.vision_transformer import vit_base
from dinov2.models.vision_transformer import vit_base_14, fix_state_dict_keys
from Gastro_IQA_model import GastroResnet

class SampleModel(pl.LightningModule):
    def __init__(self, backbone='DINOv3', model_path=None):
        super().__init__()
        self.save_hyperparameters()
        
        if backbone == 'resnet':
            path_gastro = '/home/middeljans/GastroKey/weights/checkpoint_200ep_teacher_adapted.pth'
            self.encoder = GastroResnet(path_gastro)
            self.encoder.eval()

        if backbone == 'DINOv3':
            self.encoder = vit_base(
                layerscale_init=1.0e-5,
                mask_k_bias=True,
                untie_global_and_local_cls_norm=True,
                n_storage_tokens=4,
                pos_embed_rope_rescale_coords=2,
            )
        
            if model_path:
                state_dict = torch.load(model_path, weights_only=True)
                self.encoder.load_state_dict(state_dict, strict=False)

        elif backbone == 'DINOv2':
            self.encoder = vit_base_14()
            if model_path:
                state_dict = torch.load(model_path, weights_only=True)
                state_dict = fix_state_dict_keys(state_dict)
                self.encoder.load_state_dict(state_dict, strict=False)
        
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.backbone = backbone

    def get_features(self, x):
        if self.backbone == 'resnet':
            return self.encoder(x)
        else:
            layers = self.encoder.get_intermediate_layers(x, n=1, return_class_token=True)
            return layers[-1][1] 

    def reduce_dimensions(self, x, n_components=2):
        mean = torch.mean(x, dim=0)
        x_centered = x - mean
        _, _, V = torch.pca_lowrank(x_centered, q=n_components)
        return torch.mm(x_centered, V[:, :n_components])

    def kmeans_pytorch(self, x, k, max_iters=20):
        n_samples, n_features = x.size()
        device = x.device
        indices = torch.randperm(n_samples, device=device)[:k]
        centroids = x[indices]
        
        for _ in range(max_iters):
            distances = torch.cdist(x, centroids)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(k, device=device)
            new_centroids.index_add_(0, labels, x)
            ones = torch.ones(n_samples, device=device)
            counts.index_add_(0, labels, ones)
            new_centroids /= counts.clamp(min=1).unsqueeze(1)
            if torch.allclose(centroids, new_centroids, atol=1e-4): break
            centroids = new_centroids
        return centroids

    def select_keyframes_cosine(self, feats, n_frames):
        """
        Selects frames based purely on Cosine Similarity (Diversity Maximization).
        Feats should be normalized before calling this.
        """
        T = feats.size(0)
        device = feats.device
        
        # Start with the first frame or a random frame
        selected_indices = [0] 
        
        # We track the maximum similarity of each frame to the currently selected set
        # Since we want 'most different', we want to pick the frame where this max similarity is LOWEST
        for _ in range(n_frames - 1):
            selected_feats = feats[selected_indices] # [curr_k, D]
            
            # Matrix of similarities: [T, curr_k]
            # Similarity = Dot product (because feats are normalized)
            similarities = torch.mm(feats, selected_feats.t())
            
            # For each candidate frame, find its highest similarity to any already chosen frame
            max_sims, _ = torch.max(similarities, dim=1)
            
            # Pick the frame that is LEAST similar to its closest neighbor in the selected set
            next_idx = torch.argmin(max_sims).item()
            selected_indices.append(next_idx)
            
        return torch.tensor(selected_indices, device=device, dtype=torch.long)

    def select_keyframes_cosine_threshold(self, feats, n_frames, similarity_threshold=0.5):
        """
        Selects frames based on Diversity Maximization.
        Stops if all remaining frames are more than 'similarity_threshold' 
        similar to the already selected set.
        """
        T = feats.size(0)
        device = feats.device
        
        # Start with the first frame
        selected_indices = [0] 
        
        for _ in range(n_frames - 1):
            # [curr_k, D]
            selected_feats = feats[selected_indices] 
            
            # Matrix of similarities: [T, curr_k]
            # Similarity = Dot product (assuming feats are already normalized)
            similarities = torch.mm(feats, selected_feats.t())
            
            # For each candidate frame, find its highest similarity to ANY already chosen frame
            # max_sims shape: [T]
            max_sims, _ = torch.max(similarities, dim=1)
            
            # We find the candidate that has the LOWEST 'highest similarity'
            min_max_sim = torch.min(max_sims)
            next_idx = torch.argmin(max_sims).item()

            # --- THE THRESHOLD CHECK ---
            # If the most different frame left is still > 0.5 similar to the set, 
            # then we have covered all unique semantic 'zones' in the video.
            if min_max_sim > similarity_threshold:
                # print(f"Stopping early: min-max similarity {min_max_sim:.4f} > {similarity_threshold}")
                break
                
            selected_indices.append(next_idx)
            
        return torch.tensor(selected_indices, device=device, dtype=torch.long)

    def get_embeddings(self, frames, batch_size=64, target_dim=None):
        """
        Extracts, optionally reduces, and normalizes features for all input frames.
        Returns: [T, Dim] tensor
        """
        T = frames.size(0)
        feats_list = []
        
        self.encoder.to(frames.device)
        
        for i in range(0, T, batch_size):
            batch = frames[i:i+batch_size]
            with torch.no_grad():
                f = self.get_features(batch)
            feats_list.append(f)
        
        feats = torch.cat(feats_list, dim=0)

        # Apply PCA reduction if requested
        if target_dim is not None:
            feats = self.reduce_dimensions(feats, n_components=target_dim)

        # L2 Normalize for Cosine Similarity calculations
        feats = F.normalize(feats, dim=1) 
        
        return feats

    def select_most_different_frames(self, frames, n_frames=5, batch_size=64, 
                                     use_kmeans=False, target_dim=None):
        T = frames.size(0)
        if n_frames >= T:
            return torch.arange(T, device=self.device), frames

        # 1. GPU Feature Extraction
        feats_list = []
        for i in range(0, T, batch_size):
            batch = frames[i:i+batch_size].to(self.device)
            with torch.no_grad():
                f = self.get_features(batch)
            feats_list.append(f)
        
        feats = torch.cat(feats_list, dim=0)

        # 2. PCA Reduction
        if target_dim is not None:
            feats = self.reduce_dimensions(feats, n_components=target_dim)

        # 3. Normalize for Cosine Similarity
        feats = F.normalize(feats, dim=1) 

        # 4. Selection Logic
        if use_kmeans:
            centroids = self.kmeans_pytorch(feats, k=n_frames)
            dists = torch.cdist(feats, centroids)
            indices = torch.argmin(dists, dim=0)
            # Ensure uniqueness
            # if torch.unique(indices).numel() < n_frames:
            #     indices = self.select_keyframes_cosine(feats, n_frames)
        else:
            # Pure Cosine Similarity Diversity Selection
            indices = self.select_keyframes_cosine(feats, n_frames)

        indices, _ = torch.sort(indices)
        return indices, frames[indices]
    
class SampleModel_multi(pl.LightningModule):
    def __init__(self, model_path=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = vit_base(
            layerscale_init=1.0e-5,
            mask_k_bias=True,
            untie_global_and_local_cls_norm=True,
            n_storage_tokens=4,
            pos_embed_rope_rescale_coords=2,
        )
        
        if model_path:
            state_dict = torch.load(model_path, weights_only=True)
            self.encoder.load_state_dict(state_dict, strict=False)
        
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_features(self, x):
        """
        Extracts and concatenates CLS tokens from early, middle, and late layers.
        This captures a multi-scale representation of visual difference.
        """
        target_layers = [2, 5, 8, 11]
        
        # get_intermediate_layers returns a list of tuples: [(patch_feats, cls_token), ...]
        outputs = self.encoder.get_intermediate_layers(
            x, n=target_layers, return_class_token=True
        )
        
        # 1. Extract the CLS tokens from the selected layers
        cls_tokens = [layer_output[1] for layer_output in outputs]
        
        # 2. L2-Normalize each layer's token individually. 
        # This is CRITICAL so that one layer doesn't dominate the distance calculation.
        normed_tokens = [F.normalize(t, p=2, dim=1) for t in cls_tokens]
        
        # 3. Concatenate into a single high-dimensional visual fingerprint
        fused_features = torch.cat(normed_tokens, dim=1)
        
        return fused_features 

    def reduce_dimensions(self, x, n_components=2):
        mean = torch.mean(x, dim=0)
        x_centered = x - mean
        _, _, V = torch.pca_lowrank(x_centered, q=n_components)
        return torch.mm(x_centered, V[:, :n_components])

    def kmeans_pytorch(self, x, k, max_iters=20):
        n_samples, n_features = x.size()
        device = x.device
        # Use a more stable initialization for high-dim concatenated features
        indices = torch.randperm(n_samples, device=device)[:k]
        centroids = x[indices]
        
        for _ in range(max_iters):
            distances = torch.cdist(x, centroids)
            labels = torch.argmin(distances, dim=1)
            
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(k, device=device)
            
            new_centroids.index_add_(0, labels, x)
            ones = torch.ones(n_samples, device=device)
            counts.index_add_(0, labels, ones)
            
            new_centroids /= counts.clamp(min=1).unsqueeze(1)
            
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        return centroids

    def select_keyframes_cosine(self, feats, n_frames):
        """
        Greedy Selection: Maximizes the minimum Cosine Distance between selected frames.
        """
        T = feats.size(0)
        device = feats.device
        
        # Start with the first frame of the sequence
        selected_indices = [0] 
        
        # Normalize again just to be safe for Dot Product similarity
        feats = F.normalize(feats, p=2, dim=1)
        
        for _ in range(n_frames - 1):
            selected_feats = feats[selected_indices] # [curr_k, D_total]
            
            # Matrix of similarities between all frames and currently selected frames
            # Resulting shape: [Total_Frames, Num_Selected]
            similarities = torch.mm(feats, selected_feats.t())
            
            # Find the "nearest neighbor" similarity for every frame in the video
            max_sims, _ = torch.max(similarities, dim=1)
            
            # Pick the frame whose "nearest neighbor" is the most different (lowest similarity)
            next_idx = torch.argmin(max_sims).item()
            selected_indices.append(next_idx)
            
        return torch.tensor(selected_indices, device=device, dtype=torch.long)

    def select_most_different_frames(self, frames, n_frames=5, batch_size=64, 
                                     use_kmeans=False, target_dim=None):
        T = frames.size(0)
        if n_frames >= T:
            return torch.arange(T, device=self.device), frames

        # 1. GPU Feature Extraction (using fused layers)
        feats_list = []
        for i in range(0, T, batch_size):
            batch = frames[i:i+batch_size].to(self.device)
            with torch.no_grad():
                f = self.get_features(batch)
            feats_list.append(f)
        
        feats = torch.cat(feats_list, dim=0)

        # 2. Optional PCA Reduction
        # Note: PCA might wash out the benefit of concatenation if target_dim is too low.
        if target_dim is not None:
            feats = self.reduce_dimensions(feats, n_components=target_dim)

        # 3. Final Normalization for Cosine Logic
        feats = F.normalize(feats, dim=1) 

        # 4. Selection Logic
        if use_kmeans:
            # Finding K centroids in the concatenated semantic space
            centroids = self.kmeans_pytorch(feats, k=n_frames)
            dists = torch.cdist(feats, centroids)
            indices = torch.argmin(dists, dim=0)
        else:
            # Greedy diversity selection using the fused visual fingerprint
            indices = self.select_keyframes_cosine(feats, n_frames)

        indices, _ = torch.sort(indices)
        return indices, frames[indices]