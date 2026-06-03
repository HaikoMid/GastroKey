from cv2 import threshold

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
            self.encoder = GastroResnet(model_path)
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
        
    def get_features_multi(self, x):
        """Return concatenated CLS tokens from several layers for a multi-scale representation."""
        target_layers = [2, 5, 8, 11]
        
        outputs = self.encoder.get_intermediate_layers(
            x, n=target_layers, return_class_token=True
        )
        
        # Extract CLS tokens from the chosen layers
        cls_tokens = [layer_output[1] for layer_output in outputs]

        # L2-normalize each token so no single layer dominates
        normed_tokens = [F.normalize(t, p=2, dim=1) for t in cls_tokens]

        # Concatenate into a single fused feature vector
        fused_features = torch.cat(normed_tokens, dim=1)
        
        return fused_features 

    def reduce_dimensions(self, x, n_components=2):
        mean = torch.mean(x, dim=0)
        x_centered = x - mean
        std = torch.std(x_centered, dim=0) + 1e-6
        x_scaled = x_centered / std
        _, _, V = torch.pca_lowrank(x_scaled, q=n_components)
        return torch.mm(x_scaled, V[:, :n_components])

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

    def spherical_kmeans_pytorch(self, x, k, max_iters=20):
        n_samples, n_features = x.size()
        device = x.device
        x = F.normalize(x, p=2, dim=1)
        indices = torch.randperm(n_samples, device=device)[:k]
        centroids = x[indices]
        centroids = F.normalize(centroids, p=2, dim=1)
        
        for _ in range(max_iters):
            distances = torch.cdist(x, centroids)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.zeros_like(centroids)
            new_centroids.index_add_(0, labels, x)
            counts = torch.bincount(labels, minlength=k).float().to(device)
            new_centroids /= counts.clamp(min=1).unsqueeze(1)
            new_centroids = F.normalize(new_centroids, p=2, dim=1)
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
            
        return centroids

    def select_keyframes_cosine(self, feats, n_frames):
        """Select frames by cosine similarity (diversity). Input features must be normalized."""
        T = feats.size(0)
        device = feats.device
        
        # Start with the first frame
        selected_indices = [0]

        # For each candidate, track its max similarity to the selected set;
        # choose the frame whose max similarity is smallest (most different)
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

    def select_most_different_frames(self, frames, n_frames=5, batch_size=64, multi=False,
                                     use_kmeans=False, use_spherical=False, target_dim=None):
        T = frames.size(0)
        if n_frames >= T:
            return torch.arange(T, device=self.device), frames

        # GPU Feature Extraction
        feats_list = []
        for i in range(0, T, batch_size):
            batch = frames[i:i+batch_size].to(self.device)
            with torch.no_grad():
                if multi:
                    f = self.get_features_multi(batch)
                else:
                    f = self.get_features(batch)
            feats_list.append(f)
        
        feats = torch.cat(feats_list, dim=0)

        # PCA Reduction
        if target_dim is not None:
            feats = self.reduce_dimensions(feats, n_components=target_dim)

        # Normalize for Cosine Similarity
        if not use_kmeans:
            feats = F.normalize(feats, dim=1) 

        # Selection Logic
        if use_kmeans:
            centroids = self.kmeans_pytorch(feats, k=n_frames)
            dists = torch.cdist(feats, centroids)
            indices = torch.argmin(dists, dim=0)
        elif use_spherical:
            centroids = self.spherical_kmeans_pytorch(feats, k=n_frames)
            similarities = torch.mm(feats, centroids.t())
            indices = torch.argmax(similarities, dim=0)
        else:
            indices = self.select_keyframes_cosine(feats, n_frames)

        indices, _ = torch.sort(indices)
        return indices, frames[indices]

    def _pytorch_rectangular_maxvol(self, Q, k, eps=1e-6):
        """Greedy rectangular MaxVol implemented in PyTorch."""
        T, q = Q.shape
        device = Q.device
        # Limit k to available rows/cols
        k = min(k, q, T)

        # Pick the row with largest norm as first pivot
        norms = torch.norm(Q, dim=1)
        selected_idx = [torch.argmax(norms).item()]
        remaining_mask = torch.ones(T, dtype=torch.bool, device=device)
        remaining_mask[selected_idx[0]] = False

        # Iteratively add rows that maximize the residual to the current subspace
        for _ in range(1, k):
            A_sel = Q[selected_idx]

            # SVD to get a stable basis for projections
            U, S, V = torch.linalg.svd(A_sel, full_matrices=False)

            # Project rows onto the basis and compute residual norms
            projection = (Q @ V.t()) @ V
            residuals = torch.norm(Q - projection, dim=1)

            # Ignore already selected rows
            residuals[~remaining_mask] = -1.0

            best_idx = torch.argmax(residuals).item()
            selected_idx.append(best_idx)
            remaining_mask[best_idx] = False

        return torch.tensor(sorted(selected_idx), device=device, dtype=torch.long)

    def select_most_informative_maxvol(self, frames, n_frames=5, batch_size=64,
                                    target_dim=None, adaptive=True, tol=0.1):
        """
        Optimized frame selection using PyTorch-native MaxVol logic.
        """
        T = frames.size(0)
        if n_frames >= T:
            return torch.arange(T, device=frames.device), frames

        # Extract features
        feats_list = []
        self.encoder.to(frames.device)
        
        for i in range(0, T, batch_size):
            batch = frames[i:i+batch_size]
            with torch.no_grad():
                f = self.get_features(batch)
            
            # Ensure f is a tensor and has 2 dimensions (Batch, Dim)
            if not isinstance(f, torch.Tensor):
                # If model returned a custom object/dict, try to extract the tensor
                f = f.logits if hasattr(f, 'logits') else f.last_hidden_state
                
            if f.dim() == 1:
                f = f.unsqueeze(0)
            feats_list.append(f)

        feats = torch.cat(feats_list, dim=0)

        # Dimensionality Reduction (SVD/PCA)
        if target_dim is not None or adaptive:
            dim = n_frames if adaptive else target_dim
            # PCA requires at least 'dim' rows and columns
            if feats.shape[0] > dim and feats.shape[1] > dim:
                # Re-assigning carefully to avoid 'SampleModel' assignment
                reduced_feats = self.reduce_dimensions(feats, n_components=dim)
                if isinstance(reduced_feats, torch.Tensor):
                    feats = reduced_feats
                else:
                    print("Warning: reduce_dimensions did not return a Tensor. Skipping PCA.")

        # Normalize features for MaxVol stability
        feats = F.normalize(feats, dim=1)

        # Run MaxVol
        indices = self._pytorch_rectangular_maxvol(feats, n_frames)

        # Sort indices to maintain temporal order
        indices, _ = torch.sort(indices)
        
        # Final safety slice
        indices = indices[:n_frames]

        return indices, frames[indices]