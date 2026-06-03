GastroKey — quick start

**Purpose**
- Tools to extract keyframes from endoscopy videos, build per-image cache metadata, and train downstream classification models.

**Prerequisites**
- Create a Python environment with PyTorch, torchvision, PyTorch Lightning, OpenCV, scikit-image, scipy, PIL, numba, torchmetrics and wandb. Example:

```bash
conda create -n gastro python=3.10
conda activate gastro
pip install torch torchvision pytorch-lightning opencv-python scikit-image scipy pillow numba torchmetrics wandb
```

**1) Sample frames from videos**
- Script: [GastroKey/extract_frames.py](GastroKey/extract_frames.py)
- Basic AI-assisted extraction (ROI + IQA + latent sampling). Required: `--dataset_root`, `--keyframe_dir`, and `--backbone_path` (path to pretrained backbone weights).

Example (AI pipeline):

```bash
python GastroKey/extract_frames.py \
  --dataset_root /path/to/videos \
  --keyframe_dir ./frames \
  --backbone_path ./GastroKey/weights/<model_weights>.pt \
  --target_fps 0.25 \
  --method cosine \
  --name latent \
  --target_dim 64
```

- Uniform sampling (fast, no AI):

```bash
python GastroKey/extract_frames.py \
  --dataset_root /path/to/videos \
  --keyframe_dir ./frames \
  --uniform_only \
  --target_fps 0.25
```

Notes:
- `--target_fps` controls how many frames per second are sampled for the final selection.
- `--downsample_factor` subsamples raw frames before processing (useful for long videos).
- Output frames are saved under `--keyframe_dir/<video_relative_path>/<name>/`.

**2) Build the per-image cache JSON files used by training**
- Script: [GastroKey/utils/generate_cache.py](GastroKey/utils/generate_cache.py)
- This script walks the image folders, computes ROI, assigns patient folds and writes one JSON per image.

Run from the repository root; example:

```bash
python GastroKey/utils/generate_cache.py /path/to/image/root <cache_name> <experiment_name> --splits 5 --seed 6
```

- The script writes JSON files into `./<cache_name>/<experiment_name>/` relative to where you launch it. (Run from the repo root to keep cache under the project.)

Edit paths inside and submit with `sbatch` if using a cluster.

**3) Train downstream models**
- Training entry: [GastroKey/train_wle.py](GastroKey/train_wle.py)
- The training script expects a cache directory (default used inside the script). By default it will read from `./GastroKey/<cache_name>`.

Simple example run (uses defaults set in the file; adjust flags as needed):

```bash
python GastroKey/train_wle.py --basename <CADe_model_name> --batchsize 32 --num_epochs 50 --backbone DINOv3 --weights GastroDINO --train_lr 1e-5 --method <experiment_name>
```

**Tips & troubleshooting**
- If frames or cache files are missing, check the directory structure in `--keyframe_dir` and re-run `generate_cache.py` pointing at the image root.
- For large datasets, run `extract_frames.py` per-folder or on a cluster (divide input).
- Use `--uniform_only` in `extract_frames.py` to quickly produce frames before enabling the full AI pipeline.
- If training fails looking for cache path, open `GastroKey/train_wle.py` and confirm `CACHE_PATH`/`SAVE_DIR` match your layout.

**Files of interest**
- [GastroKey/extract_frames.py](GastroKey/extract_frames.py) — frame extraction and keyframe selection
- [GastroKey/utils/generate_cache.py](GastroKey/utils/generate_cache.py) — build per-image JSON cache
- [GastroKey/train_wle.py](GastroKey/train_wle.py) — training script (PyTorch Lightning)
