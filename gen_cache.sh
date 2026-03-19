#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00

# activate conda environment
source activate intern

python utils/generate_cache.py