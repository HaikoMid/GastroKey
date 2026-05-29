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
        """
        Extracts and concatenates CLS tokens from early, middle, and late layers.
        This captures a multi-scale representation of visual difference.
        """
        target_layers = [2, 5, 8, 11]
        
        outputs = self.encoder.get_intermediate_layers(
            x, n=target_layers, return_class_token=True
        )
        
        # 1. Extract the CLS tokens from the selected layers
        cls_tokens = [layer_output[1] for layer_output in outputs]
        
        # 2. L2-Normalize each layer's token individually. 
        # This is so that one layer doesn't dominate the distance calculation.
        normed_tokens = [F.normalize(t, p=2, dim=1) for t in cls_tokens]
        
        # 3. Concatenate into a single high-dimensional visual fingerprint
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

    def select_most_different_frames(self, frames, n_frames=5, batch_size=64, multi=False,
                                     use_kmeans=False, use_spherical=False, target_dim=None):
        T = frames.size(0)
        if n_frames >= T:
            return torch.arange(T, device=self.device), frames

        # 1. GPU Feature Extraction
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

        # 2. PCA Reduction
        if target_dim is not None:
            feats = self.reduce_dimensions(feats, n_components=target_dim)

        # 3. Normalize for Cosine Similarity
        if not use_kmeans:
            feats = F.normalize(feats, dim=1) 

        # 4. Selection Logic
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
    
    # def _rectangular_maxvol(self, Q, k, max_iters=20, eps=1e-6, tol=1e-6):
    #     """
    #     Approximate rectangular MaxVol selection on rows of Q (T x q).
    #     Returns indices of k rows whose submatrix maximizes rect-vol ≈ det(A A^T).

    #     This implementation uses a greedy forward selection followed by
    #     pairwise swap improvements. It's not the most optimal/fast MaxVol
    #     implementation, but it's stable and simple to integrate.
    #     """
    #     T, q = Q.shape

    #     # If requested k is larger than q, warn and cap k to q to avoid
    #     # singular Gram matrices (determinant will be zero otherwise).
    #     if k > q:
    #         k = q

    #     device = Q.device

    #     # Greedy forward selection: at each step pick the row that maximizes
    #     # the log-determinant of the Gram matrix of the selected set.
    #     selected = []
    #     remaining = list(range(T))

    #     for _ in range(k):
    #         best_idx = None
    #         best_ld = -float('inf')

    #         for idx in remaining:
    #             cand = selected + [idx]
    #             A = Q[cand]  # [m, q]
    #             G = A @ A.t()
    #             # regularize for numerical stability
    #             G = G + eps * torch.eye(G.shape[0], device=device)
    #             sign, ld = torch.linalg.slogdet(G)
    #             if sign <= 0:
    #                 ld_val = -float('inf')
    #             else:
    #                 ld_val = ld.item()

    #             if ld_val > best_ld:
    #                 best_ld = ld_val
    #                 best_idx = idx

    #         if best_idx is None:
    #             break

    #         selected.append(best_idx)
    #         remaining.remove(best_idx)

    #     # Local pairwise swap improvements
    #     improved = True
    #     it = 0
    #     while improved and it < max_iters:
    #         improved = False
    #         it += 1

    #         A_sel = Q[selected]
    #         G_sel = A_sel @ A_sel.t() + eps * torch.eye(len(selected), device=device)
    #         sign_sel, ld_sel = torch.linalg.slogdet(G_sel)
    #         base_ld = ld_sel.item() if sign_sel > 0 else -float('inf')

    #         best_swap = None
    #         best_ld = base_ld

    #         for i, s_idx in enumerate(selected):
    #             for c_idx in remaining:
    #                 cand = list(selected)
    #                 cand[i] = c_idx
    #                 A = Q[cand]
    #                 G = A @ A.t() + eps * torch.eye(len(cand), device=device)
    #                 sign, ld = torch.linalg.slogdet(G)
    #                 ld_val = ld.item() if sign > 0 else -float('inf')
    #                 if ld_val > best_ld + tol:
    #                     best_ld = ld_val
    #                     best_swap = (i, s_idx, c_idx)

    #         if best_swap is not None:
    #             i, s_idx, c_idx = best_swap
    #             # perform swap
    #             remaining.remove(c_idx)
    #             remaining.append(s_idx)
    #             selected[i] = c_idx
    #             improved = True

    #     return torch.tensor(sorted(selected), device=device, dtype=torch.long)

    # def select_most_informative_maxvol(self, frames, n_frames=5, batch_size=64,
    #                                    use_pca=True, target_dim=None, max_iters=20, adaptive=True):
    #     """
    #     Selects the most informative frames using a rectangular MaxVol-style
    #     selection on the feature matrix Q (rows = frames, cols = features).

    #     - Extracts features for all frames (GPU batched)
    #     - Optionally reduces dimensionality via PCA (recommended when
    #       features_dim > n_frames)
    #     - Runs an approximate MaxVol selection to pick `n_frames` rows
    #       whose submatrix maximizes rect-vol (det(A A^T)).

    #     Returns (indices, frames[indices]).
    #     """
    #     T = frames.size(0)
    #     if n_frames >= T:
    #         return torch.arange(T, device=frames.device), frames

    #     # 1. Extract features
    #     feats_list = []
    #     self.encoder.to(frames.device)
    #     for i in range(0, T, batch_size):
    #         batch = frames[i:i+batch_size].to(self.device)
    #         with torch.no_grad():
    #             f = self.get_features(batch)
    #         # if model returns batch x D, append per-row
    #         if f.dim() == 1:
    #             f = f.unsqueeze(0)
    #         feats_list.append(f.cpu())

    #     feats = torch.cat(feats_list, dim=0).to(frames.device)

    #     # 2. Optional PCA reduction to ensure enough columns (q) for maxvol
    #     if target_dim is not None:
    #         if adaptive:
    #             feats = self.reduce_dimensions(feats, n_components=n_frames)
    #         else:
    #             feats = self.reduce_dimensions(feats, n_components=target_dim)

    #     # If number of columns < n_frames, we cannot get a full-rank Gram matrix.
    #     # Cap n_frames to number of columns.
    #     if n_frames > feats.shape[1]:
    #         n_frames = feats.shape[1]

    #     # Normalize to stabilize numeric range
    #     feats = F.normalize(feats, dim=1)

    #     # 3. Run MaxVol helper
    #     indices = self._rectangular_maxvol(feats, n_frames, max_iters=max_iters)

    #     return indices, frames[indices]

    def _pytorch_rectangular_maxvol(self, Q, k, eps=1e-6):
        """
        Pure PyTorch implementation of Greedy Rectangular MaxVol.
        Uses vectorized batching for speed.
        """
        T, q = Q.shape
        device = Q.device
        # Safety: Cannot select more frames than available rows or columns
        k = min(k, q, T)
        
        # 1. Start with the row of largest norm as the first pivot
        norms = torch.norm(Q, dim=1)
        selected_idx = [torch.argmax(norms).item()]
        remaining_mask = torch.ones(T, dtype=torch.bool, device=device)
        remaining_mask[selected_idx[0]] = False

        # 2. Iteratively add rows
        for _ in range(1, k):
            # Current selection submatrix
            A_sel = Q[selected_idx]
            
            # Use SVD to find the basis of the currently selected subspace
            # This is more stable than slogdet for greedy selection
            U, S, V = torch.linalg.svd(A_sel, full_matrices=False)
            
            # Project all rows onto the current basis
            # V.t() are the principal components
            projection = (Q @ V.t()) @ V
            residuals = torch.norm(Q - projection, dim=1)
            
            # Mask out already selected rows by setting residual to -1
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

        # 1. Extract features
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

        # 2. Dimensionality Reduction (SVD/PCA)
        # Only run if we have enough samples to satisfy the target dimension
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

        # 3. Run MaxVol helper
        # We call self._pytorch_rectangular_maxvol because it's a class method
        indices = self._pytorch_rectangular_maxvol(feats, n_frames)

        # Sort indices to maintain temporal order
        indices, _ = torch.sort(indices)
        
        # Final safety slice
        indices = indices[:n_frames]

        return indices, frames[indices]