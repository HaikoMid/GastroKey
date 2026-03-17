## !/bin/bash

## Set job requirements
## SBATCH --nodes=1
## SBATCH --gpus-per-node=1
## SBATCH --cpus-per-task=16
## SBATCH --partition=gpu_h100
## SBATCH --time=12:00:00

## activate conda environment
## module load 2023
## module load Anaconda3/2023.07-2
## source activate intern

## module load 2023
## module load CUDA/12.4.0
## cd /gpfs/home4/middeljans/segdino/dinov3/eval/segmentation/models/utils/ops
## rm -rf build
## python setup.py build_ext --inplace
## export PYTHONPATH=/gpfs/home4/middeljans/segdino/dinov3/eval/segmentation/models/utils/ops:$PYTHONPATH

# python train_wle.py --scope Fuji Pentax Olympus --basename CaFormer_images_damper --train_lr 0.000001 --backbone CaFormer-S18 --weights GastroNet --task cls --type images --damper

# python train_wle.py --scope Fuji Pentax Olympus --basename Vit-b14_336_10ps --train_lr 0.00001 --backbone Vit-b14 --task cls --type images --imagesize 336 --damper --num_epochs 20 --image_perc 10
# python train_wle.py --scope Fuji Pentax Olympus --basename Vit-b16_dinov3_Gastro_336_10ps --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --imagesize 336 --damper --num_epochs 20 --image_perc 10
# python train_wle.py --scope Fuji Pentax Olympus --basename Vit-b16_dinov3_Gastro_336_10ps_hlr --train_lr 0.0001 --backbone Vit-b16 --task cls --type images --imagesize 336 --damper --num_epochs 20 --image_perc 10
# python train_wle.py --scope Fuji Pentax Olympus --basename Dinov3_336_10ps --train_lr 0.00001 --backbone dinov3 --task cls --type images --imagesize 336 --damper --num_epochs 20 --image_perc 10

# python train_wle.py --scope Fuji Pentax Olympus --basename GastroNet_5ps_llr_s11 --train_lr 0.000003 --backbone Vit-b14 --task cls --type images --imagesize 336 --damper --num_epochs 30 --image_perc 5
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_5ps_mlr_s11 --train_lr 0.00003 --backbone Vit-b16 --task cls --type images --imagesize 336 --damper --num_epochs 30 --image_perc 5
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroNet_5ps_hlr_ --train_lr 0.00001 --backbone Vit-b14 --task cls --type images --imagesize 336 --damper --num_epochs 30 --image_perc 5


# python train_wle.py --scope Fuji Pentax Olympus --basename DinoV3_5ps_mlr --train_lr 0.00003 --backbone dinov3 --task cls --type images --imagesize 336 --damper --num_epochs 30 --image_perc 5
# python train_wle.py --scope Fuji Pentax Olympus --basename DinoV3_10ps_mlr --train_lr 0.00003 --backbone dinov3 --task cls --type images --imagesize 336 --damper --num_epochs 30 --image_perc 10

# python train_wle.py --scope Fuji Pentax Olympus --basename DinoV3_5ps --train_lr 0.00001 --backbone dinov3 --task cls --type images --imagesize 336 --damper --num_epochs 30 --image_perc 5
# python train_wle.py --scope Fuji Pentax Olympus --basename DinoV3_100ps --train_lr 0.00001 --backbone dinov3 --task cls --type images --imagesize 336 --damper --num_epochs 30 --image_perc 100

# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_resonly_256 --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --imagesize 256 --damper --num_epochs 30 --weights res_only
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_resonly_512 --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --imagesize 512 --damper --num_epochs 10 --weights res_only

# python train_wle.py --scope Fuji Pentax Olympus --basename ResNet_images_GastroNet --train_lr 0.00001 --backbone ResNet-50-GastroNet --task cls --type images

# python train_wle.py --scope Fuji --basename Fuji_images --train_lr 0.00001 --task cls --type images
# python train_wle.py --scope Pentax --basename Pentax_images --train_lr 0.00001 --task cls --type images
# python train_wle.py --scope Olympus --basename Olympus_images --train_lr 0.00001 --task cls --type images

# python train_wle.py --scope Fuji Pentax Olympus --basename ResNet_images --train_lr 0.00001 --task cls --type images

# python inference_wle.py

# python train_wle.py --scope Pentax --basename Pentax_frames_100_bs64 --train_lr 0.00001 --backbone ResNet-50-ImageNet --task cls --type images frames --frame_perc 100 --batchsize 64
# python train_wle.py --scope Pentax --basename Pentax_frames_50 --train_lr 0.00001 --backbone ResNet-50-ImageNet --task cls --type images frames --frame_perc 50
# python train_wle.py --scope Pentax --basename Pentax_frames_10 --train_lr 0.00001 --backbone ResNet-50-ImageNet --task cls --type images frames --frame_perc 10

# python train_wle.py --scope Pentax --basename Pentax_frames_100 --train_lr 0.00001 --backbone ResNet-50-ImageNet --task cls --type images frames --frame_perc 100
# python train_wle.py --scope Pentax --basename Pentax_frames_50 --train_lr 0.00001 --backbone ResNet-50-ImageNet --task cls --type images frames --frame_perc 50
# python train_wle.py --scope Pentax --basename Pentax_frames_10 --train_lr 0.00001 --backbone ResNet-50-ImageNet --task cls --type images frames --frame_perc 10

# python train_wle.py --scope Olympus --basename Olympus_frames_100_bs128 --train_lr 0.00001 --backbone ResNet-50-ImageNet --task cls --type images frames --frame_perc 100 --batchsize 128
# python train_wle.py --scope Olympus --basename Olympus_frames_50 --train_lr 0.00001 --backbone ResNet-50-ImageNet --task cls --type images frames --frame_perc 50
# python train_wle.py --scope Olympus --basename Olympus_frames_10 --train_lr 0.00001 --backbone ResNet-50-ImageNet --task cls --type images frames --frame_perc 10

# python train_wle.py --scope Fuji Pentax Olympus --basename DinoV3_llr_s8 --train_lr 0.00001 --backbone dinov3 --task cls --type images --imagesize 336 --damper --num_epochs 15 --image_perc 100 --seed 7
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_llr_s8 --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --imagesize 336 --damper --num_epochs 15 --image_perc 100 --seed 7
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroNet_llr_s8 --train_lr 0.00001 --backbone Vit-b14 --task cls --type images --imagesize 336 --damper --num_epochs 15 --image_perc 100 --seed 7

# python train_wle.py --scope Fuji Pentax Olympus --basename DinoV3_25ps_llr_s8 --train_lr 0.00001 --backbone dinov3 --task cls --type images --imagesize 336 --damper --num_epochs 25 --image_perc 25 --seed 7
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_25ps_llr_s8 --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --imagesize 336 --damper --num_epochs 25 --image_perc 25 --seed 7
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroNet_25ps_llr_s8 --train_lr 0.00001 --backbone Vit-b14 --task cls --type images --imagesize 336 --damper --num_epochs 25 --image_perc 25 --seed 7

# python train_wle.py --scope Fuji Pentax Olympus --basename DinoV3_5ps_llr_s8 --train_lr 0.00001 --backbone dinov3 --task cls --type images --imagesize 336 --damper --num_epochs 25 --image_perc 5 --seed 7
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_5ps_llr_s8 --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --imagesize 336 --damper --num_epochs 25 --image_perc 5 --seed 7
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroNet_5ps_llr_s8 --train_lr 0.00001 --backbone Vit-b14 --task cls --type images --imagesize 336 --damper --num_epochs 25 --image_perc 5 --seed 7

# python train_wle.py --scope Fuji Pentax Olympus --basename GastroNet_5ps_sllr_s7 --train_lr 0.000003 --backbone Vit-b14 --task cls --type images --imagesize 336 --damper --num_epochs 25 --image_perc 5 --seed 7
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroNet_5ps_sllr_s8 --train_lr 0.000003 --backbone Vit-b14 --task cls --type images --imagesize 336 --damper --num_epochs 25 --image_perc 5 --seed 8

# python train_wle.py --scope Fuji Pentax Olympus --basename Multihead --train_lr 0.00001 --backbone Vit-b16 --task cls_multihead --type images --damper --seed 7
# python train_wle.py --scope Fuji --basename Fuji --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --damper --seed 7
# python train_wle.py --scope Pentax --basename Pentax --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --damper --seed 7
# python train_wle.py --scope Olympus --basename Olympus --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --damper --seed 7

# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_frames10_ex --train_lr 0.00001 --backbone Vit-b16 --task cls --type images frames --imagesize 336 --damper --num_epochs 10 --frame_perc 10 --seed 7 --exclusive_frames
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_frames25_ex --train_lr 0.00001 --backbone Vit-b16 --task cls --type images frames --imagesize 336 --damper --num_epochs 10 --frame_perc 25 --seed 7 --exclusive_frames
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_frames100_ex --train_lr 0.00001 --backbone Vit-b16 --task cls --type images frames --imagesize 336 --damper --num_epochs 10 --frame_perc 100 --seed 7 --exclusive_frames
# python train_wle.py --scope Fuji Pentax Olympus --basename GastroDINO_512_s7 --train_lr 0.00001 --backbone Vit-b16 --task cls --type images --imagesize 512 --damper --num_epochs 15 --image_perc 100 --seed 7

# python train_wle.py --scope Fuji Olympus --basename Seg_GastroDINO_s9_WLI --train_lr 0.00001 --backbone Vit-b16 --task seg --imagesize 336 --damper --num_epochs 50 --seed 9 --weights res --modality WLI --freeze_backbone --decoder upernet
# python train_wle.py --scope Fuji Olympus --basename Seg_GastroNET_s9_WLI --train_lr 0.00001 --backbone Vit-b14 --task seg --imagesize 336 --damper --num_epochs 50 --seed 9 --weights res --modality WLI --freeze_backbone --decoder upernet
# python train_wle.py --scope Fuji Olympus --basename Seg_DINOv3_s9_WLI --train_lr 0.00001 --backbone DinoV3 --task seg --imagesize 336 --damper --num_epochs 50 --seed 9 --modality WLI --freeze_backbone --decoder upernet

# python train_wle.py --scope Fuji Olympus --basename Seg_GastroDINO_s9_p25_WLI --train_lr 0.00001 --backbone Vit-b16 --task seg --imagesize 336 --damper --num_epochs 50 --seed 9 --weights res --modality WLI --freeze_backbone --decoder upernet --image_perc 25
# python train_wle.py --scope Fuji Olympus --basename Seg_GastroNET_s9_p25_WLI --train_lr 0.00001 --backbone Vit-b14 --task seg --imagesize 336 --damper --num_epochs 50 --seed 9 --weights res --modality WLI --freeze_backbone --decoder upernet --image_perc 25
# python train_wle.py --scope Fuji Olympus --basename Seg_DINOv3_s9_p25_WLI --train_lr 0.00001 --backbone DinoV3 --task seg --imagesize 336 --damper --num_epochs 50 --seed 9 --modality WLI --freeze_backbone --decoder upernet --image_perc 25

# python train_wle.py --scope Fuji Olympus --basename Seg_GastroDINO_s9_p5_WLI --train_lr 0.00001 --backbone Vit-b16 --task seg --imagesize 336 --damper --num_epochs 50 --seed 9 --weights res --modality WLI --freeze_backbone --decoder upernet --image_perc 5
# python train_wle.py --scope Fuji Olympus --basename Seg_GastroNET_s9_p5_WLI --train_lr 0.00001 --backbone Vit-b14 --task seg --imagesize 336 --damper --num_epochs 50 --seed 9 --weights res --modality WLI --freeze_backbone --decoder upernet --image_perc 5
# python train_wle.py --scope Fuji Olympus --basename Seg_DINOv3_s9_p5_WLI --train_lr 0.00001 --backbone DinoV3 --task seg --imagesize 336 --damper --num_epochs 50 --seed 9 --modality WLI --freeze_backbone --decoder upernet --image_perc 5

# python train_wle.py --scope Fuji Pentax Olympus --basename multihead --train_lr 0.00001 --backbone Vit-b16 --task combi_multihead --type images --imagesize 336 --num_epochs 10 --weights last --decoder dpt --weight_sampling class --damper

python train_wle.py --scope Fuji Pentax Olympus --basename fold0_multihead --train_lr 0.00001 --backbone Vit-b16 --task combi_multihead --type images --imagesize 256 --num_epochs 8 --weights last --decoder dpt --weight_sampling class --val_fold 0

python train_wle.py --scope Fuji --basename fold0_Fuji --train_lr 0.00001 --backbone Vit-b16 --task combi --type images --imagesize 256 --num_epochs 8 --weights last --decoder dpt --weight_sampling class --val_fold 0
python train_wle.py --scope Pentax --basename fold0_Pentax --train_lr 0.00001 --backbone Vit-b16 --task combi --type images --imagesize 256 --num_epochs 8 --weights last --decoder dpt --weight_sampling class --val_fold 0
python train_wle.py --scope Olympus --basename fold0_Olympus --train_lr 0.00001 --backbone Vit-b16 --task combi --type images --imagesize 256 --num_epochs 8 --weights last --decoder dpt --weight_sampling class --val_fold 0

python train_wle.py --scope Fuji Pentax Olympus --basename fold0_last --train_lr 0.00001 --backbone Vit-b16 --task combi --type images --imagesize 256 --num_epochs 8 --weights last --decoder dpt --weight_sampling class --val_fold 0
python train_wle.py --scope Fuji Pentax Olympus --basename fold0_120k --train_lr 0.00001 --backbone Vit-b16 --task combi --type images --imagesize 256 --num_epochs 8 --weights 120k --decoder dpt --weight_sampling class --val_fold 0