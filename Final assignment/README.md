# SegFormer for Autonomous Driving: Balancing Real-Time Efficiency and Minority Class Safety

**Author:** Josha Kuipers  
**TU/e Email:** j.j.j.kuipers@student.tue.nl  
**Codalab Usernames:**
- Peak Performance: `JK_URBAN_SF_B1_Adaptive`, `JK_URBAN_SF_B1_Custom`, `JK_URBAN_SF_B1_Blank`
- Efficiency: `JK_HIGHWAY_SF_B0_Adaptive`, `JK_HIGHWAY_SF_B0_Custom`, `JK_HIGHWAY_SF_B0_Blank`
- Baseline: `JK_Segf_CS_Baseline_V3`
---

## Overview

This repository contains the full training, evaluation, and benchmarking pipeline for semantic segmentation on the Cityscapes dataset, developed for the 5LSM0 Neural Networks for Computer Vision final assignment. The project investigates the SegFormer architecture with a focus on two competing demands: real-time inference for autonomous driving deployment, and accurate detection of safety-critical minority classes such as pedestrians and cyclists.

Standard segmentation models tend to bias toward dominant background classes (road, sky, buildings), which make up roughly 80% of Cityscapes pixels. This pipeline addresses that imbalance through targeted transfer learning strategies and Weighted Cross-Entropy loss, while enforcing strict FPS constraints derived from vehicle velocity physics.

Two benchmarks are targeted:

- **Peak Performance** — SegFormer-B1 at 384x768, targeting urban environments (>= 30 FPS)
- **Efficiency** — SegFormer-B0 at 384x768, targeting high-speed highway scenarios (>= 60 FPS)

---

## Related Guides

This repository includes several additional guides for specific workflows:

- `README-Installation.md` — Setting up VSCode, GitHub, Docker, W&B, and MobaXTerm from scratch
- `README-Slurm.md` — Cloning the repo on the HPC cluster, downloading data and the container, and submitting training jobs via SLURM
- `README-Submission.md` — Building, testing, and exporting a Docker image for the challenge server
- `README-Report.md` — Guidelines for writing and structuring the research paper

---

## Repository Structure

```
.
├── config.py                   # Central configuration: model size, resolution, weight init
├── model.py                    # SegFormer architecture with dynamic backbone freezing
├── model_Unet.py               # U-Net baseline architecture
├── train.py                    # Training loop with W&B logging and class-weighted loss
├── predict.py                  # Inference script for the challenge evaluation server
├── benchmark_fps.py            # CUDA event-based FPS benchmarking at FP32 precision
├── main.sh                     # Training command passed to the container on the cluster
├── jobscript_slurm.sh          # Parameters about Snellius to train
├── Dockerfile                  # Container for reproducible server submission
├── download_docker_and_data.sh # SLURM script to download the container and dataset
├── README-Installation.md      # Tool installation and environment setup guide
├── README-Slurm.md             # HPC cluster and SLURM job submission guide
├── README-Submission.md        # Docker build, test, and challenge submission guide
└── README-Report.md            # Research paper writing guidelines
```

---

## Quick Start

There are two ways to run this project: locally on your own machine, or on the HPC cluster via SLURM. Both ultimately use the same `train.py` and `config.py`.

### Option A — Local

**1. Install dependencies**

Python 3.8 or higher is required.

```bash
pip install torch torchvision transformers wandb numpy Pillow
```

A CUDA-capable GPU is required for training and benchmarking. All experiments were run on an NVIDIA GeForce RTX 2060 (6 GB VRAM) with PyTorch 2.10.0 and CUDA 12.8.

**2. Download the Cityscapes dataset**

Download the dataset from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) and place it at `./data/cityscapes`.

**3. Configure the run**

Open `config.py` and set your desired track before running any script (see the Configuration section below).

**4. Train**

```bash
wandb login
python train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 50 \
    --lr 0.00006 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer-b1-384x768-cityscapes"
```

Use `--disable-wandb` to skip W&B logging during local testing.

### Option B — HPC Cluster (SLURM)

See `README-Slurm.md` for the full cluster workflow. The short version:

**1. Clone your fork on the cluster and run the one-time download script:**

```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```

This pulls the course Apptainer container (`container.sif`) and downloads the Cityscapes dataset from Hugging Face into `./data/`.

**2. Set your W&B credentials in `.env`:**

```env
WANDB_API_KEY=your_key_here
WANDB_DIR=/home/<username>/wandb
```

**3. Edit `main.sh` with your training command, then submit:**

```bash
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

---

## Configuration

All settings are controlled from a single file: `config.py`. Set the following variables before running any script.

```python
# Resolution
HEIGHT = 384    # 256 for Efficiency Track, 384 for Peak Performance
WIDTH  = 768    # 512 for Efficiency Track, 768 for Peak Performance

# Architecture
MODEL_SIZE = "b1"   # "b0" (~3.7M params) or "b1" (~13.7M params)

# Weight initialization strategy
INIT_WEIGHTS = "cityscapes"   # "cityscapes" or "blank"
```

### Weight Initialization Strategies

**`INIT_WEIGHTS = "cityscapes"`** (Adaptive configuration)
- Loads a fully pre-trained Cityscapes SegFormer checkpoint. Automatically tries `/app/segformer-b{0,1}-cityscapes` first, then falls back to Hugging Face.
- Automatically freezes the backbone to preserve optimized spatial features.
- Uses a filtered AdamW optimizer that only updates the classification head.
- Recommended when fine-tuning with a custom loss function without disrupting learned representations.

**`INIT_WEIGHTS = "blank"`** (Blank / Custom configuration)
- Loads ImageNet pre-trained MiT backbone weights. Automatically tries `/app/mit-b{0,1}` first, then falls back to Hugging Face.
- Leaves the full backbone unfrozen.
- Uses a differential AdamW optimizer: the classification head trains at the standard learning rate, the backbone at `LR * 0.01`.
- Recommended for training class-imbalance weighting from scratch without gradient shock.

---

## Training Details

The training loop (`train.py`) tracks the following metrics on the Cityscapes validation split:
- Mean IoU and Mean Dice (all 19 classes)
- Human IoU and Human Dice (Person, Rider)
- Vehicle IoU and Vehicle Dice (Car, Truck, Bus, Train, Motorcycle, Bicycle)

Model checkpoints are saved to `checkpoints/<experiment-id>/`. The best checkpoint by validation loss is kept and updated throughout training. A final checkpoint is always saved at the end of the last epoch.

### Loss Function

Three loss function options are available in `train.py`. Switch between them by commenting and uncommenting the relevant block around line 262.

**Option 1: Weighted Cross-Entropy (default, recommended)**

```python
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
```

The primary loss function used in all final submissions. Applies per-class multipliers to penalize misclassification of minority classes more heavily than dominant background classes.

**Option 2: Weighted Focal Loss**

```python
criterion = WeightedFocalLoss(weight=class_weights, gamma=2.0, ignore_index=255)
```

The `WeightedFocalLoss` class is defined at the top of `train.py`. It combines the same class weights with the focal modulation term `(1 - pt)^gamma`, which further down-weights easy, confidently-classified pixels. In practice this converged faster but produced lower final validation scores and proved less stable during fine-tuning.

**Option 3: Standard Cross-Entropy (no weighting)**

```python
criterion = nn.CrossEntropyLoss(ignore_index=255)
```

Used for the unweighted Blank baseline experiments. Without class weights the model tends to over-predict dominant background classes.

**Class weights applied in options 1 and 2:**

| Train ID | Class | Weight |
|----------|-------|--------|
| 0 | road | 0.5 (suppress) |
| 1 | sidewalk | 0.8 (suppress) |
| 2 | building | 0.5 (suppress) |
| 3 | wall | 1.0 |
| 4 | fence | 1.2 |
| 5 | pole | 1.2 |
| 6 | traffic light | 1.5 |
| 7 | traffic sign | 1.5 |
| 8 | vegetation | 0.5 (suppress) |
| 9 | terrain | 1.0 |
| 10 | sky | 0.5 (suppress) |
| 11 | person | 3.0 (max boost) |
| 12 | rider | 3.0 (max boost) |
| 13 | car | 1.2 |
| 14 | truck | 2.0 |
| 15 | bus | 2.0 |
| 16 | train | 2.0 |
| 17 | motorcycle | 3.0 (max boost) |
| 18 | bicycle | 3.0 (max boost) |

The 3.0x ceiling was chosen based on Phan and Yamamoto (2020) to avoid gradient instability. A pure inverse-frequency weight for the rarest classes would exceed 30x, which causes training divergence.

---

## Benchmarking Inference Speed

To verify real-time compliance before submission, run the FPS benchmark:

```bash
python benchmark_fps.py
```

This script uses `torch.cuda.Event` GPU-side timestamps, as prescribed by the PyTorch CUDA documentation, to avoid measuring only kernel launch time. It performs 50 warm-up iterations followed by 500 timed forward passes at FP32 precision, batch size 1, and reports mean latency, standard deviation, and FPS.

Place test images in `../local_data/` or update `IMAGE_DIR` at the top of the script. If no images are found, a random dummy tensor is used automatically.

---

## Docker: Building and Submitting

The `Dockerfile` provides a fully self-contained environment for inference and challenge submission. It downloads all four model checkpoints (MiT-B0, MiT-B1, SegFormer-B0-Cityscapes, SegFormer-B1-Cityscapes) at build time, so no internet access is required at runtime.

For the full submission workflow — including how to copy your checkpoint, build the image, test it locally, export it as a `.tar`, and upload to the challenge servers — see `README-Submission.md`.

**Short version:**

```bash
# 1. Copy your best checkpoint
cp checkpoints/<your-experiment>/best_model-*.pt ./model.pt

# 2. Build the image
docker build -t nncv-submission:latest -f Dockerfile .

# 3. Test locally
docker run --rm \
  -v "$(pwd)/local_data:/data" \
  -v "$(pwd)/local_output:/output" \
  nncv-submission:latest

# 4. Export for submission
docker save -o nncv_submission.tar nncv-submission:latest
```

The container runs `predict.py` automatically as its entrypoint. It reads all `.png` files from `/data` and writes predicted segmentation masks to `/output`.

---

## Results Summary

All results below are from the official evaluation server on the unseen test set.

**Peak Performance (SegFormer-B1, 384x768)**

|Codalab Usernames| Configuration | mIoU | Human IoU | Vehicle IoU |
|---|---|---|---|---|
|JK_Segf_CS_Baseline_V3| U-Net Baseline | 0.4048 | 0.1006 | 0.1791 |
|JK_URBAN_SF_B1_Blank| B1 Blank, Unweighted | 0.3933 | 0.1146 | 0.1912 |
|JK_URBAN_SF_B1_Custom| B1 Custom (Blank + WCE) | 0.4203 | 0.1712 | 0.2350 |
|JK_URBAN_SF_B1_Adaptive| B1 Adaptive (Pre-trained) | 0.4512 | 0.1791 | 0.2745 |

**Efficiency (SegFormer-B0, 256x512)**

|Codalab Usernames| Configuration | mIoU | FPS | GFLOPs | TFLOPs |
|---|---|---|---|---|---|
|JK_Segf_CS_Baseline_V3| U-Net Baseline | 0.4007 | 4.09 | 2563.5 | 0.1913 |
|JK_HIGHWAY_SF_B0_Blank| B0 Blank, Unweighted | 0.3743 | 16.29 | 241.9 | 1.9176 |
|JK_HIGHWAY_SF_B0_Custom| B0 Custom (Blank + WCE) | 0.4027 | 16.31 | 241.9 | 2.0637 |
|JK_HIGHWAY_SF_B0_Adaptive| B0 Adaptive (Pre-trained) | 0.4342 | 16.29 | 241.9 | 2.1814 |

Local benchmarks (RTX 2060, FP32, batch size 1) confirmed B1 at 384x768 achieves 43 FPS and B0 at 384x768 achieves 63 FPS, satisfying the 30 FPS urban and 60 FPS highway deployment thresholds. The lower FPS reported by the evaluation server reflects the overhead of its GFLOP profiling hooks and does not represent real-world inference speed.

---

## Acknowledgments

- SegFormer architecture: Xie et al. (2021), NVIDIA / Hugging Face Transformers
- Cityscapes dataset: Cordts et al. (2016)
- Weighted Cross-Entropy methodology: Phan and Yamamoto (2020)
- FPS benchmarking approach: PyTorch CUDA documentation and benchmark tutorial