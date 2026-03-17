#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --time=06:30:00

# activate conda environment
module load 2023
module load Anaconda3/2023.07-2
source activate intern

python extract_frames.py