import os
import time
import torch
from PIL import Image
import torchvision.transforms as T
# from model_UNET import Model
from model import Model
from config import IMG_SIZE

def load_real_images(image_dir, resolution, device):
    """Loads images from a folder, resizes, normalizes, and moves them to the GPU."""
    supported_formats = ('.png', '.jpg', '.jpeg')
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(supported_formats)]
    
    if not image_paths:
        print(f"Warning: No images found in {image_dir}. Falling back to dummy data.")
        return [torch.randn(1, 3, resolution[0], resolution[1]).to(device)]

    print(f"Loaded {len(image_paths)} real images from {image_dir}.")
    
    # Preprocessing to match SegFormer expectations
    transform = T.Compose([
        T.Resize(resolution),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # For SegFormer
        # T.Normalize(mean=(0.5,), std=(0.5,)) # For U-Net, adjust if you changed the normalization in predict.py
    ])

    preloaded_images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension & move to GPU
        preloaded_images.append(tensor)
        
    return preloaded_images

def measure_fps(resolution, weights_path=None, image_dir=None, use_fp16=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize and Load Model
    model = Model(n_classes=19)
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded weights from: {weights_path}")
        except Exception as e:
            print(f"Error loading weights (Ensure it's a state_dict, not a full model): {e}")
            return
    else:
        print("Warning: No valid weights path provided. Testing with random weights.")
            
    model = model.to(device)
    model.eval()

    # 2. Load Real Images BEFORE timing
    if image_dir and os.path.exists(image_dir):
        test_images = load_real_images(image_dir, resolution, device)
    else:
        print(f"Warning: Directory {image_dir} not found. Using dummy data.")
        test_images = [torch.randn(1, 3, resolution[0], resolution[1]).to(device)]

    num_images = len(test_images)
    iterations = 100

    # 3. Warm-up Phase
    with torch.no_grad():
        with torch.autocast(device_type=device, enabled=use_fp16):
            for i in range(10):
                _ = model(test_images[i % num_images])

    # 4. Timing Phase
    if device == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                for i in range(iterations):
                    _ = model(test_images[i % num_images])
                    
        ender.record()
        torch.cuda.synchronize() 
        total_time_seconds = starter.elapsed_time(ender) / 1000.0 
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            for i in range(iterations):
                _ = model(test_images[i % num_images])
        total_time_seconds = time.perf_counter() - start_time

    # 5. Calculate Results
    fps = iterations / total_time_seconds
    mode = "FP16" if use_fp16 and device == "cuda" else "FP32"
    print(f"Resolution: {resolution[0]}x{resolution[1]} | Precision: {mode} | FPS: {fps:.2f}\n")

if __name__ == "__main__":
    # --- Exact paths based on your terminal layout ---
    # Since you run this from inside "Final assignment", model.pt is in the same folder
    my_model_file = "model.pt" 
    
    # local_data is one directory up
    my_image_folder = "../local_data" 
    
    print("--- Testing FPS with Trained Model and Real Images ---")
    
    # Test 1: Your target resolution (FP32)
    measure_fps(
        resolution=(IMG_SIZE), 
        weights_path=my_model_file, 
        image_dir=my_image_folder,
        use_fp16=False 
    )

    # Test 2: Your target resolution (FP16 - highly recommended for speed!)
    measure_fps(
        resolution=(IMG_SIZE), 
        weights_path=my_model_file, 
        image_dir=my_image_folder,
        use_fp16=True 
    )