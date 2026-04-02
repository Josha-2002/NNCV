"""
benchmark_fps.py — FPS & Latency Benchmarking for SegFormer Semantic Segmentation

Methodology:
  - Device:       GPU preferred (CUDA Events for timing), CPU fallback (perf_counter)
  - Precision:    FP32 (submission standard) and optionally FP16
  - Batch size:   1 (real-time autonomous driving condition)
  - Warm-up:      50 iterations with explicit CUDA sync before timing starts
  - Timing:       500 iterations; reports mean latency, std dev, and FPS
  - Input:        Real images from disk (fallback: synthetic dummy tensor)

All reported FPS values are based on FP32 unless explicitly noted as FP16.
FP16 is NOT the submission evaluation condition (per config.py).
"""

import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from model import Model
from config import IMG_SIZE

# ── Benchmark hyperparameters ────────────────────────────────────────────────
WARMUP_RUNS    = 50
TIMING_RUNS    = 500
# ─────────────────────────────────────────────────────────────────────────────


def print_environment(device: str):
    """Prints hardware and software environment for report reproducibility."""
    print("=" * 60)
    print("BENCHMARK ENVIRONMENT")
    print("=" * 60)
    print(f"  PyTorch version : {torch.__version__}")
    if device == "cuda":
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory      : {mem_gb:.1f} GB")
    else:
        print("  Device          : CPU (no CUDA GPU detected)")
    print(f"  IMG_SIZE        : {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print("=" * 60)
    print()


def load_real_images(image_dir: str, resolution: tuple, device: str) -> list:
    """
    Loads images from a folder, applies SegFormer preprocessing, and moves
    them to the target device. Falls back to a single dummy tensor if the
    folder is empty or missing.
    """
    if not image_dir or not os.path.exists(image_dir):
        print(f"  Warning: '{image_dir}' not found. Falling back to dummy data.")
        return [torch.randn(1, 3, resolution[0], resolution[1]).to(device)]

    supported = ('.png', '.jpg', '.jpeg')
    paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(supported)
    ]

    if not paths:
        print(f"  Warning: No images in '{image_dir}'. Falling back to dummy data.")
        return [torch.randn(1, 3, resolution[0], resolution[1]).to(device)]

    transform = T.Compose([
        T.Resize(resolution),
        T.ToTensor(),
        # ImageNet normalization — matches predict.py and train.py
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensors = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        tensors.append(transform(img).unsqueeze(0).to(device))

    print(f"  Loaded {len(tensors)} real image(s) from '{image_dir}'.")
    return tensors


def measure_fps(
    resolution: tuple,
    weights_path: str = None,
    image_dir: str    = None,
    use_fp16: bool    = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Environment info ──────────────────────────────────────────────────
    print_environment(device)

    # ── 2. Build label for report ────────────────────────────────────────────
    precision_label = "FP16 ⚠️  (NOT submission condition)" if (use_fp16 and device == "cuda") else "FP32 (submission standard)"

    # ── 3. Load model ────────────────────────────────────────────────────────
    model = Model(n_classes=19)

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(
            weights_path,
            map_location=device,
            weights_only=False,   # Required for HuggingFace SegFormer objects
        )
        model.load_state_dict(state_dict, strict=False)
        print(f"  Weights loaded : {weights_path}")
    else:
        print("  ⚠️  No valid weights path. Benchmarking with random weights.")

    model = model.to(device).eval()

    # ── 4. Pre-load images ───────────────────────────────────────────────────
    images  = load_real_images(image_dir, resolution, device)
    n_imgs  = len(images)
    print()

    # ── 5. Warm-up ───────────────────────────────────────────────────────────
    print(f"  Running {WARMUP_RUNS} warm-up iterations...")
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(use_fp16 and device == "cuda")):
            for i in range(WARMUP_RUNS):
                _ = model(images[i % n_imgs])

    # Flush GPU pipeline completely before timing starts
    if device == "cuda":
        torch.cuda.synchronize()

    # ── 6. Timed benchmark ───────────────────────────────────────────────────
    print(f"  Running {TIMING_RUNS} timed iterations...")
    latencies_ms = []

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(use_fp16 and device == "cuda")):

            if device == "cuda":
                for i in range(TIMING_RUNS):
                    starter = torch.cuda.Event(enable_timing=True)
                    ender   = torch.cuda.Event(enable_timing=True)

                    # Sync BEFORE recording start — guarantees clean slate
                    torch.cuda.synchronize()
                    starter.record()

                    _ = model(images[i % n_imgs])

                    ender.record()
                    torch.cuda.synchronize()   # Wait for this iteration to finish

                    latencies_ms.append(starter.elapsed_time(ender))

            else:
                for i in range(TIMING_RUNS):
                    t0 = time.perf_counter()
                    _ = model(images[i % n_imgs])
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    # ── 7. Report ────────────────────────────────────────────────────────────
    arr          = np.array(latencies_ms)
    mean_ms      = arr.mean()
    std_ms       = arr.std()
    fps          = 1000.0 / mean_ms
    fps_worst    = 1000.0 / arr.max()
    fps_best     = 1000.0 / arr.min()
    realtime_ok  = fps >= 30.0

    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Resolution      : {resolution[0]}x{resolution[1]}")
    print(f"  Precision       : {precision_label}")
    print(f"  Iterations      : {TIMING_RUNS}")
    print(f"  Mean latency    : {mean_ms:.2f} ms")
    print(f"  Std deviation   : {std_ms:.2f} ms  (lower = more stable)")
    print(f"  Mean FPS        : {fps:.2f}")
    print(f"  FPS range       : {fps_worst:.2f} (worst) – {fps_best:.2f} (best)")
    print(f"  Real-time (≥30) : {'✅ YES' if realtime_ok else '❌ NO'}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    my_model_file  = "model.pt"
    my_image_folder = "../local_data"

    print("\n Test 1 — FP32 (Submission Standard)\n")
    measure_fps(
        resolution   = IMG_SIZE,
        weights_path = my_model_file,
        image_dir    = my_image_folder,
        use_fp16     = False,
    )

    print("\n Test 2 — FP16 (Speed Reference Only, NOT submission condition)\n")
    measure_fps(
        resolution   = IMG_SIZE,
        weights_path = my_model_file,
        image_dir    = my_image_folder,
        use_fp16     = True,
    )