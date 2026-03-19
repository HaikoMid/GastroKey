import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from algorithms import SampleModel, SampleModel_multi
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
    parser.add_argument('--backbone', type=str, default='DINOv3')
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

def biq_frames(model_input_tensor, model, threshold=2.0, batch_size=64):
    """
    Optimized: Now accepts a pre-loaded model instead of loading weights every time.
    """
    dataloader = DataLoader(model_input_tensor, batch_size=batch_size, shuffle=False)
    iqa_list = []

    with torch.no_grad():
        for data in dataloader:
            y_iqa, _, _, _ = model(data)
            iqa_list.append(y_iqa.detach().cpu())

    scores = torch.cat(iqa_list, dim=0).squeeze().numpy()
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

def process_video(video_path, opt, device, iqa_model, sample_model):
    """
    Optimized: Accepts pre-loaded models and saves to a flat directory.
    """
    model_inputs, raw_frames, global_indices, original_fps = extract_frames_from_video(
        str(video_path), opt.downsample_factor, opt.imagesize
    )
    
    if model_inputs is None: return

    duration = (len(raw_frames) * opt.downsample_factor) / original_fps
    n_target = 10 #int(max(1, round(duration * opt.target_fps)))
    
    # Define the SINGLE flat output directory
    save_dir = Path(opt.keyframe_dir) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    video_stem = video_path.stem # e.g., 'Patient_01_Video_A'

    # 1. Quality Filter (using pre-loaded model)
    valid_subset_idx = biq_frames(model_inputs, iqa_model, threshold=opt.quality_threshold)
    
    # 2. Fallback Logic
    if len(valid_subset_idx) == 0:
        fallback_threshold = 1.0
        print(f"No frames passed {opt.quality_threshold} for {video_path.stem}. Retrying with threshold {fallback_threshold}...")
        
        valid_subset_idx = biq_frames(model_inputs, iqa_model, threshold=fallback_threshold)
        
        if len(valid_subset_idx) == 0:
            print(f"Skipping {video_path.name}: Even with fallback, no frames passed quality check.")
            return
    
    actual_n = min(n_target, len(valid_subset_idx))
    
    if actual_n == 0:
        print(f"Skipping {video_path.name}: No frames passed quality threshold.")
        return

    # 3. Latent Sampling
    hq_inputs = model_inputs[valid_subset_idx]
    with torch.no_grad():
        ai_sub_idx, _ = sample_model.select_most_different_frames(
            hq_inputs, n_frames=actual_n, use_kmeans=opt.use_kmeans, target_dim=opt.target_dim
        )

    # 4. Save to Flat Folder with Unique Filenames
    for i, sub_idx in enumerate(ai_sub_idx):
        idx_in_hq = sub_idx.item()
        orig_idx = global_indices[valid_subset_idx[idx_in_hq]]
        frame_bgr = cv2.cvtColor(raw_frames[valid_subset_idx[idx_in_hq]], cv2.COLOR_RGB2BGR)
        
        # Filename includes video name to prevent overwriting
        out_name = f"{video_stem}_f{orig_idx:06d}.png"
        cv2.imwrite(str(save_dir / out_name), frame_bgr)

def run_uniform_extraction(video_path, opt):
    """
    Flattened version of uniform extraction.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return

    n_target = int(max(1, round((total_frames / fps) * opt.target_fps)))
    save_dir = Path(opt.keyframe_dir) / "uniform"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.linspace(0, total_frames - 1, num=n_target, dtype=int)
    video_stem = video_path.stem

    for i, target_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if ret:
            out_name = f"{video_stem}_uni{i:02d}_fr{target_idx:06d}.png"
            cv2.imwrite(str(save_dir / out_name), frame)
    cap.release()

def main():
    opt = get_params()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- LOAD MODELS ONCE ---
    print("Loading AI Models into memory...")
    path_gastro = '/home/middeljans/GastroKey/weights/checkpoint_200ep_teacher_adapted.pth'
    path_iqa_weights = '/home/middeljans/GastroKey/weights/model_iqa_eso_clean.pt'
    
    # IQA Model
    iqa_model = GastroIQA_multihead(path_gastro).to(device).eval()
    iqa_model.load_state_dict(torch.load(path_iqa_weights, weights_only=True))
    
    # Sampling Model
    sample_model = SampleModel_multi(model_path=opt.backbone_path).to(device).eval()

    root_path = Path(opt.dataset_root)
    all_videos = list(root_path.rglob('*.[mM][pP]4'))
    video_files = all_videos
    
    print(f"Total videos to process: {len(video_files)}")
    
    for video_path in video_files:
        try:
            print(f"\n--- Processing: {video_path.name} ---")
            if opt.uniform_only:
                run_uniform_extraction(video_path, opt)
            else:
                process_video(video_path, opt, device, iqa_model, sample_model)
        except Exception as e:
            print(f"Failed to process {video_path.name}: {e}")

if __name__ == '__main__':
    main()