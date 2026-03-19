#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --time=12:00:00

# activate conda environment
module load 2023
module load Anaconda3/2023.07-2
source activate intern

# python cosine.py --name test --backbone DINOv3 --backbone_path '/home/middeljans/COSMO/pretrained/Gastro231k.pth'


# python extract_frames.py --name latent_DINOv2 --backbone DINOv2 --backbone_path '/home/middeljans/COSMO/pretrained/dinov2_base.pth' --imagesize 336
# python extract_frames.py --name latent_resnet --backbone resnet

python extract_frames_resnet.py --name frames --backbone resnet --use_kmeans --target_dim 2 --dataset_root /projects/0/prjs1485/videos_ndbe --keyframe_dir "/home/middeljans/Video summarization/sampled_frames_ndbe" --quality_threshold 2 --downsample_factor 2