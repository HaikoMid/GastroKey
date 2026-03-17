import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from algorithms import SampleModel
from torchvision import transforms
from torchvision.models import resnet50
from scipy import ndimage
from skimage.measure import label
from pathlib import Path

WLE_MEAN = [0.64041256, 0.36125767, 0.31330117]
WLE_STD = [0.18983584, 0.15554344, 0.14093774]

def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--imagesize', type=int, default=256)
    parser.add_argument('--n_keyframes', type=int, default=5)
    parser.add_argument('--keyframe_dir', type=str, default='/projects/0/prjs1485/GastroKey')
    parser.add_argument('--keyframe_batch_size', type=int, default=64)
    parser.add_argument('--dataset_root', type=str, default='/projects/0/prjs1485/1. Datasets Cosmo/')
    parser.add_argument('--downsample_factor', type=int, default=1)
    parser.add_argument('--target_dim', type=int, default=None)
    parser.add_argument('--quality_threshold', type=float, default=2.0)
    parser.add_argument('--use_kmeans', action='store_true', help="Whether to use KMeans for frame selection instead of pure cosine similarity.")
    parser.add_argument('--name', type=str, default='latent')
    parser.add_argument('--backbone_path', type=str, default='/home/middeljans/COSMO/pretrained/Gastro231k.pth') #/home/middeljans/COSMO/pretrained/dinov3_vitb16_pretrain_lvd1689m.pth
    parser.add_argument('--target_fps', type=float, default=0.25, help='Number of frames to extract per second of video duration')
    parser.add_argument('--uniform_only', action='store_true', help='If set, only sample uniform frames and skip all AI/Quality processing.')
    return parser.parse_args()

class GastroIQA_multihead(torch.nn.Module):
    def __init__(self, GastroWeights):
        super(GastroIQA_multihead, self).__init__()
        self.weights = torch.load(GastroWeights, weights_only=True)
        self.resnet = torch.nn.Sequential()
        [self.resnet.add_module(name, child) for name, child in resnet50(weights=None).named_children() if
         name != 'fc']
        self.resnet.load_state_dict(self.weights, strict=True)
        # create one head
        self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.iqa_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 1)
        )
        self.esophagus_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 1)
        )

        self.cleaning_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 1)
        )

    def forward(self, x):
        y = self.resnet(x)

        #one head
        y = self.avg_pooling(y)

        #iqa
        y_iqa = self.iqa_layers(y)

        #esophagus
        y_eso = self.esophagus_layers(y)

        #cleaning
        y_cleaning = self.cleaning_layers(y)


        return y_iqa, y_eso, y_cleaning, y

def biq_frames(model_input_tensor, original_indices, threshold=2.0, batch_size=64):
    """
    Filters the model_input_tensor based on GastroIQA scores.
    """
    device = model_input_tensor.device
    path_gastro = '/home/middeljans/GastroKey/weights/checkpoint_200ep_teacher_adapted.pth'
    
    model = GastroIQA_multihead(path_gastro).to(device)
    model.eval()

    path_iqa= '/home/middeljans/GastroKey/weights/model_iqa_eso_clean.pt'
    iqa_weights = torch.load(path_iqa, weights_only=True)
    model.load_state_dict(iqa_weights, strict=True)

    dataloader = DataLoader(model_input_tensor, batch_size=batch_size, shuffle=False)
    iqa_list = []

    with torch.no_grad():
        for data in dataloader:
            y_iqa, _, _, y = model(data)
            print(f"Batch IQA Scores: {y_iqa.squeeze().cpu().numpy()}")
            iqa_list.append(y_iqa.detach().cpu())

    scores = torch.cat(iqa_list, dim=0).squeeze().numpy()
    
    # Identify which frames in this batch pass the test
    valid_mask = scores >= threshold
    valid_indices = np.where(valid_mask)[0]

    print(f"Quality Filter: {len(valid_indices)}/{len(scores)} frames passed threshold {threshold}")
    
    return valid_indices

# Define function for minimum pooling the images
def min_pooling(img, g=8):
    # Copy Image
    out = img.copy()
    # Determine image shape and compute step size for pooling
    h, w = img.shape
    nh = int(h / g)
    nw = int(w / g)
    # Perform minimum pooling
    for y in range(nh):
        for x in range(nw):
            out[g * y:g * (y + 1), g * x:g * (x + 1)] = np.min(out[g * y:g * (y + 1), g * x:g * (x + 1)])
    return out


# Define function for finding largest connected region in images
def getlargestcc(segmentation):
    # Use built-in label method, to label connected regions of an integer array
    labels = label(segmentation)
    # Assume at least 1 CC
    assert (labels.max() != 0)  # assume at least 1 CC
    # Find the largest of connected regions, return as True and False
    largestcc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestcc


# Define function for finding bounding box coordinates for ROI in images
def bbox(img):
    # Find rows and columns where a True Bool is encountered
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    # Find the first and last row/column for the bounding box coordinates
    # cmin = left border, cmax = right border, rmin = top border, rmax = bottom border
    # Usage Image.crop((left, top, right, bottom))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


# Define Complete function for finding ROI bounding box coordinates, by combining previous functions
def find_roi(img):
    # Open image as numpy array
    image = np.array(img, dtype=np.uint8)
    # Compute L1 norm of the image
    norm_org = np.linalg.norm(image, axis=-1)
    # Use Gaussian Filter to capture low-frequency information
    img_gauss = ndimage.gaussian_filter(norm_org, sigma=5)
    # Scale pixel values
    img_scaled = ((norm_org - np.min(img_gauss)) / (np.max(img_gauss) - np.min(img_gauss))) * 255
    # Use minimum pooling
    img_norm = min_pooling(img_scaled, g=8)
    # Find largest connected region with threshold image as input
    th = 10
    largestcc = getlargestcc(img_norm >= th)
    # Obtain cropping coordinates
    rmin, rmax, cmin, cmax = bbox(largestcc)
    return rmin, rmax, cmin, cmax

def select_uniform_frames(frames, n_frames=5):
    """
    Selects n_frames evenly spaced throughout the sequence.
    """
    T = len(frames)
    if n_frames >= T:
        return torch.arange(T), frames

    # Calculate indices: e.g., if T=100 and n=5, indices = [0, 25, 50, 75, 99]
    # np.linspace handles the math perfectly
    indices = np.linspace(0, T - 1, num=n_frames, dtype=int)
    
    # If input is a list (like raw_frames), we index it with a loop or list comp
    if isinstance(frames, list):
        selected_frames = [frames[i] for i in indices]
    else:
        # If input is a torch tensor
        selected_frames = frames[indices]
        
    return torch.tensor(indices, dtype=torch.long), selected_frames

def extract_frames_from_video(video_path, downsample_factor=1, imagesize=256):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames in video file: {total_video_frames}")
    
    raw_frames = [] 
    processed_frames = []
    global_indices = [] # Track original frame numbers
    idx = 0
    roi_coords = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = torch.tensor(WLE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(WLE_STD, device=device).view(1, 3, 1, 1)

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        if (idx % downsample_factor) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if roi_coords is None: roi_coords = find_roi(frame_rgb)
            
            raw_frames.append(frame_rgb)
            global_indices.append(idx) # Save the EXACT frame number
            
            rmin, rmax, cmin, cmax = roi_coords
            crop = frame_rgb[rmin:rmax, cmin:cmax]
            resized = cv2.resize(crop, (imagesize, imagesize))
            processed_frames.append(torch.from_numpy(resized))
        idx += 1
    cap.release()

    if not processed_frames: return None, [], []

    model_input_tensor = torch.stack(processed_frames).to(device).permute(0, 3, 1, 2).float() / 255.0
    model_input_tensor = (model_input_tensor - mean) / std

    return model_input_tensor, raw_frames, global_indices, video_fps

def process_video(video_path, opt, device):
    # 1. Extraction
    # Note: Added video_fps return to extract_frames_from_video as discussed previously
    model_inputs, raw_frames, global_indices, original_fps = extract_frames_from_video(
        str(video_path), opt.downsample_factor, opt.imagesize
    )
    
    if model_inputs is None: return

    # Calculate target N
    duration = (len(raw_frames) * opt.downsample_factor) / original_fps
    n_target = int(max(1, round(duration * opt.target_fps)))
    
    # Create unique output folder for this video
    video_rel_path = video_path.relative_to(opt.dataset_root).with_suffix('')
    save_base = Path(opt.keyframe_dir) / video_rel_path
    
    # --- LOGIC BRANCH: UNIFORM ONLY ---
    if opt.uniform_only:
        print(f"Uniform-Only Mode: Sampling {n_target} frames for {video_path.name}")
        uni_dir = save_base / 'uniform'
        uni_dir.mkdir(parents=True, exist_ok=True)
        
        indices = np.linspace(0, len(raw_frames) - 1, num=n_target, dtype=int)
        for i, idx in enumerate(indices):
            frame_bgr = cv2.cvtColor(raw_frames[idx], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(uni_dir / f"uni_{i:02d}_fr{global_indices[idx]:06d}.png"), frame_bgr)
        return # Skip the rest

    # --- FULL PIPELINE (Default) ---
    # 2. Quality Filter
    valid_subset_idx = biq_frames(model_inputs, global_indices, threshold=opt.quality_threshold)
    actual_n = min(n_target, len(valid_subset_idx))
    
    if actual_n == 0:
        print(f"Skipping {video_path.name}: No frames passed quality threshold.")
        return

    # 3. Latent Sampling
    hq_inputs = model_inputs[valid_subset_idx]
    sample_model = SampleModel(model_path=opt.backbone_path).to(device)
    
    with torch.no_grad():
        ai_sub_idx, _ = sample_model.select_most_different_frames(hq_inputs, n_frames=actual_n)

    # 4. Save Latent Only
    latent_dir = save_base / 'latent'
    latent_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sub_idx in enumerate(ai_sub_idx):
        idx_in_hq = sub_idx.item()
        orig_idx = global_indices[valid_subset_idx[idx_in_hq]]
        frame_bgr = cv2.cvtColor(raw_frames[valid_subset_idx[idx_in_hq]], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(latent_dir / f"kf_{i:02d}_fr{orig_idx:06d}.png"), frame_bgr)

# def run_combi(opt):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if opt.video_path is not None:
    #     # 1. Extraction (Downsampled)
    #     model_inputs, raw_frames, global_indices, original_fps = extract_frames_from_video(
    #         opt.video_path, opt.downsample_factor, opt.imagesize
    #     )

    #     total_frames_count = len(raw_frames) * opt.downsample_factor
    #     duration_seconds = total_frames_count / original_fps
    #     n_to_sample = int(max(1, round(duration_seconds * opt.target_fps)))
    #     print(f"Duration: {duration_seconds:.2f}s | Target FPS: {opt.target_fps}")
    #     print(f"Targeting {n_to_sample} total keyframes.")

    #     # 2. Quality Filtering (Thresholding)
    #     valid_subset_idx = biq_frames(model_inputs, global_indices, threshold=opt.quality_threshold)
        
    #     actual_n = min(n_to_sample, len(valid_subset_idx))
    #     if actual_n < n_to_sample:
    #         print(f"Warning: Only {len(valid_subset_idx)} frames passed quality. Sampling all of them.")

    #     # Create high-quality subsets
    #     hq_model_inputs = model_inputs[valid_subset_idx]
    #     hq_raw_frames = [raw_frames[i] for i in valid_subset_idx]
    #     hq_global_indices = [global_indices[i] for i in valid_subset_idx]

    #     # 3. Setup AI Model & Sample from HQ subset
    #     sample_model = SampleModel(model_path=opt.backbone_path).to(device)
    #     sample_model.eval()

    #     with torch.no_grad():
    #         # AI Selection
    #         sub_ai_idx, _ = sample_model.select_most_different_frames(
    #             hq_model_inputs, n_frames=actual_n, use_kmeans=opt.use_kmeans, target_dim=2
    #         )
            
    #         uni_indices = np.linspace(0, len(raw_frames) - 1, num=actual_n, dtype=int)

    #     # 4. Save Logic
    #     ai_out_dir = os.path.join(opt.keyframe_dir, opt.name)
    #     uni_out_dir = os.path.join(opt.keyframe_dir, 'uniform')
    #     os.makedirs(ai_out_dir, exist_ok=True)
    #     os.makedirs(uni_out_dir, exist_ok=True)

    #     # Save AI (Latent)
    #     for i, sub_idx in enumerate(sub_ai_idx):
    #         idx_val = sub_idx.item()
    #         real_frame_num = hq_global_indices[idx_val] # Map back to original video index
    #         frame_bgr = cv2.cvtColor(hq_raw_frames[idx_val], cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(os.path.join(ai_out_dir, f'{i:02d}_idx{real_frame_num:06d}.png'), frame_bgr)

    #     # Save Uniform
    #     for i, idx_val in enumerate(uni_indices):
    #         real_frame_num = global_indices[idx_val]
    #         frame_bgr = cv2.cvtColor(raw_frames[idx_val], cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(os.path.join(uni_out_dir, f'{i:02d}_idx{real_frame_num:06d}.png'), frame_bgr)

    #     print(f"Done. AI frames used {len(hq_model_inputs)} high-quality candidates.")

def run_uniform_extraction(video_path, opt):
    """
    Fast extraction that skips all AI processing.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0: return
    
    # Calculate target number of frames
    duration = total_frames / fps
    n_target = int(max(1, round(duration * opt.target_fps)))
    
    # Create output directory
    video_rel_path = video_path.relative_to(opt.dataset_root).with_suffix('')
    save_dir = Path(opt.keyframe_dir) / video_rel_path / 'uniform'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate uniform indices
    indices = np.linspace(0, total_frames - 1, num=n_target, dtype=int)
    
    # Extract only what we need
    for i, target_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if ret:
            fname = f"uni_{i:02d}_fr{target_idx:06d}.png"
            cv2.imwrite(str(save_dir / fname), frame)
            
    cap.release()
    print(f"Uniform mode: Saved {len(indices)} frames to {save_dir}")

def main():
    opt = get_params()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    root_path = Path(opt.dataset_root)
    
    # 1. Find all mp4 files recursively
    all_videos = list(root_path.rglob('*.[mM][pP]4'))
    
    # 2. Filter for "Training set" only
    video_files = [v for v in all_videos if "Training set" in v.parts]
    
    print(f"Total videos found: {len(all_videos)}")
    print(f"Videos in 'Training set': {len(video_files)}")
    
    if len(video_files) == 0:
        print("No videos found. Check if 'Training set' is spelled exactly in your path.")
        return
    
    for video_path in video_files:
        try:
            print(f"\n--- Processing: {video_path.name} ---")
            
            # --- UNIFORM ONLY OPTIMIZATION ---
            if opt.uniform_only:
                # In uniform mode, we don't need ROI, IQA, or Latent models.
                # We can use a simplified version of extraction.
                run_uniform_extraction(video_path, opt)
            else:
                # Run the full AI-pipeline (ROI + IQA + Latent)
                process_video(video_path, opt, device)
                
        except Exception as e:
            print(f"Failed to process {video_path.name}: {e}")

if __name__ == '__main__':
    main()