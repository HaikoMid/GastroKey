import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from dinov3.models.vision_transformer import vit_base

class SampleModel(pl.LightningModule):
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

    def select_most_different_frames(self, frames, n_frames=5, batch_size=64, 
                                     use_kmeans=False, target_dim=2):
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