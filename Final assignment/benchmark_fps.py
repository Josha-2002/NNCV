"""
benchmark_fps_minimal.py
========================
FPS benchmark for SegFormer semantic segmentation inference.

SOURCES — every pattern in this file is traceable to one of:
  [1] PyTorch CUDA Semantics documentation
      https://docs.pytorch.org/docs/stable/notes/cuda.html
      → CUDA Event timing pattern (lines marked SOURCE-1)

  [2] PyTorch Benchmark tutorial
      https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html
      → warm-up requirement, synchronization requirement (lines marked SOURCE-2)
"""

import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from model import Model # SegFormer implementation; replace with your model if needed
# from model_Unet import Model # UNet alternative
from config import IMG_SIZE

# ── Settings ─────────────────────────────────────────────────────────────────
WARMUP_RUNS = 50   # SOURCE-2: warm-up required before timing
TIMING_RUNS = 500
MODEL_PATH  = "model.pt"
IMAGE_DIR   = "../local_data"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU       : {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"PyTorch   : {torch.__version__}")
    print(f"CUDA      : {torch.version.cuda if device == 'cuda' else 'N/A'}")
    print(f"Precision : FP32")
    print(f"IMG_SIZE  : {IMG_SIZE}")
    print()

    # ── Load model────────────────────────────────
    model = Model(n_classes=19).to(device).eval()
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded from {MODEL_PATH}")
    else:
        print("Warning: no model.pt found, using random weights.")

    # ── Load images ───────────────────────────────
    # ImageNet normalization matches predict.py
    transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    paths = [
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ] if os.path.exists(IMAGE_DIR) else []

    if paths:
        images = [transform(Image.open(p).convert('RGB')).unsqueeze(0).to(device) for p in paths]
        print(f"Loaded {len(images)} image(s) from {IMAGE_DIR}")
    else:
        images = [torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)]
        print("Warning: no images found, using dummy tensor.")
    print()

    # ── Warm-up (SOURCE-2) ───────────────────────────────────────────────────
    # PyTorch benchmark tutorial: warm-up is required so that CUDA kernel
    # JIT compilation and cuBLAS initialisation do not skew timing results.
    with torch.no_grad():
        for i in range(WARMUP_RUNS):
            _ = model(images[i % len(images)])

    # Flush GPU pipeline before timing starts (SOURCE-1)
    if device == "cuda":
        torch.cuda.synchronize()

    # ── Timed loop (SOURCE-1) ────────────────────────────────────────────────
    # Pattern taken directly from PyTorch CUDA semantics docs:
    #   start_event = torch.cuda.Event(enable_timing=True)
    #   end_event   = torch.cuda.Event(enable_timing=True)
    #   start_event.record()
    #   # ... work ...
    #   end_event.record()
    #   torch.cuda.synchronize()
    #   elapsed_ms = start_event.elapsed_time(end_event)
    latencies_ms = []
    with torch.no_grad():
        for i in range(TIMING_RUNS):
            if device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()          # SOURCE-1: sync before start
                start_event.record()              # SOURCE-1
                _ = model(images[i % len(images)])
                end_event.record()                # SOURCE-1
                torch.cuda.synchronize()          # SOURCE-1: wait for GPU
                latencies_ms.append(start_event.elapsed_time(end_event))  # SOURCE-1
            else:
                t0 = time.perf_counter()
                _ = model(images[i % len(images)])
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    # ── Results (ADDITION: statistics not in sources) ────────────────────────
    arr     = np.array(latencies_ms)
    mean_ms = arr.mean()
    std_ms  = arr.std()
    fps     = 1000.0 / mean_ms

    print("=" * 45)
    print("RESULTS (FP32, batch size 1)")
    print("=" * 45)
    print(f"  Iterations   : {TIMING_RUNS}")
    print(f"  Mean latency : {mean_ms:.2f} ms")
    print(f"  Std deviation: {std_ms:.2f} ms")
    print(f"  FPS          : {fps:.2f}")
    print(f"  Real-time    : {'YES (>=30 FPS)' if fps >= 30 else 'NO (<30 FPS)'}")
    print("=" * 45)


if __name__ == "__main__":
    main()