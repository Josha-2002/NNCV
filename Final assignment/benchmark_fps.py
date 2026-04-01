import time
import torch
from model import Model # Assuming your Model class is here

def measure_fps(resolution, use_fp16=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Model(n_classes=19).to(device)
    model.eval()

    # Create a fake image matching your exact resolution
    dummy_input = torch.randn(1, 3, resolution[0], resolution[1]).to(device)

    # 1. Warm-up
    with torch.no_grad():
        with torch.autocast(device_type=device, enabled=use_fp16):
            for _ in range(10):
                _ = model(dummy_input)

    iterations = 100

    # 2. Timing block (Handles both CPU and GPU)
    if device == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                for _ in range(iterations):
                    _ = model(dummy_input)
                    
        ender.record()
        torch.cuda.synchronize() 
        total_time_seconds = starter.elapsed_time(ender) / 1000.0 
    else:
        print("Warning: Testing on CPU. Results will be slow and less precise.")
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)
                
        total_time_seconds = time.perf_counter() - start_time

    # 3. Calculate and Print
    fps = iterations / total_time_seconds
    mode = "FP16" if use_fp16 and device == "cuda" else "FP32"
    print(f"Resolution: {resolution[0]}x{resolution[1]} | Precision: {mode} | FPS: {fps:.2f}")

if __name__ == "__main__":
    print("--- Running FPS Benchmark ---")
    
    # Standard FP32 tests
    measure_fps(resolution=(256, 512))   
    measure_fps(resolution=(384, 768))   
    measure_fps(resolution=(512, 1024))  

    print("\n--- Running High-Speed (FP16) Benchmark ---")
    # FP16 (Mixed Precision) tests - highly recommended for SegFormer
    measure_fps(resolution=(256, 512), use_fp16=True)   
    measure_fps(resolution=(512, 1024), use_fp16=True)


# import torch
# from model import Model

# def measure_fps(resolution):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if device == "cpu":
#         print("Warning: Testing on CPU. Results will be very slow!")
        
#     model = Model(n_classes=19).to(device)
#     model.eval()

#     # Create a fake image matching your exact resolution
#     dummy_input = torch.randn(1, 3, resolution[0], resolution[1]).to(device)

#     # 1. Warm-up (The first few passes on a GPU are always slow)
#     with torch.no_grad():
#         for _ in range(10):
#             _ = model(dummy_input)

#     # 2. Setup precise CUDA timers
#     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#     iterations = 100
    
#     # 3. Measure
#     starter.record()
#     with torch.no_grad():
#         for _ in range(iterations):
#             _ = model(dummy_input)
#     ender.record()
    
#     # 4. Wait for GPU to finish all tasks, then calculate
#     torch.cuda.synchronize() 
#     total_time_seconds = starter.elapsed_time(ender) / 1000.0 
    
#     fps = iterations / total_time_seconds
#     print(f"Resolution: {resolution[0]}x{resolution[1]} | FPS: {fps:.2f}")

# if __name__ == "__main__":
#     print("--- Running FPS Benchmark ---")
#     measure_fps(resolution=(256, 512))   # Your fast Efficiency size
#     measure_fps(resolution=(384, 768))   # Your fast Efficiency size
#     measure_fps(resolution=(512, 1024))  # Your Peak Performance size