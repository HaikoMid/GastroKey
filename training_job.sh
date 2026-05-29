#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --time=8:00:00

activate conda environment
module load 2023
module load Anaconda3/2023.07-2
source activate intern

# python utils/remove_cache.py
# python utils/remove_data.py
python extract_frames.py --name latent_spherical_025 --backbone DINOv3 --backbone_path '/home/middeljans/COSMO/pretrained/Gastro231k.pth' --imagesize 256 --method spherical --target_fps 0.25
python utils/generate_cache.py /projects/0/prjs1485/GastroKey /home/middeljans/GastroKey/cache latent_spherical_025
python utils/rebase_cache.py
python train_wle.py --basename latent_spherical_025 --train_lr 0.00001 --backbone DINOv3 --weights GastroDINO --task cls --imagesize 256 --num_epochs 10 --weight_sampling class --val_fold 0 --method latent_spherical_025
python inference_wle.py --names latent_spherical_025

python extract_frames.py --name latent_spherical_05 --backbone DINOv3 --backbone_path '/home/middeljans/COSMO/pretrained/Gastro231k.pth' --imagesize 256 --method spherical --target_fps 0.5
python utils/generate_cache.py /projects/0/prjs1485/GastroKey /home/middeljans/GastroKey/cache latent_spherical_05
python utils/rebase_cache.py
python train_wle.py --basename latent_spherical_05 --train_lr 0.00001 --backbone DINOv3 --weights GastroDINO --task cls --imagesize 256 --num_epochs 10 --weight_sampling class --val_fold 0 --method latent_spherical_05
python inference_wle.py --names latent_spherical_05