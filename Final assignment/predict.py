"""
This script provides and example implementation of a prediction pipeline 
for a PyTorch U-Net model. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

from model import Model
from config import IMG_SIZE
# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

#can be customized
def preprocess(img: Image.Image) -> torch.Tensor:
    # SegFormer/ImageNet standard normalization
    transform = Compose([
        ToImage(),
        # Resize(size=(256, 512), interpolation=InterpolationMode.BILINEAR), # <--- CHANGED!
        Resize(IMG_SIZE, interpolation=InterpolationMode.BILINEAR), # <--- ORIGINAL SIZE! Adjust this if you changed the image resize above.
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #For segformer, used ImageNet normalization. 
        # Normalize(mean=(0.5,), std=(0.5,)), # For U-Net, used simple normalization to [0,1]. Adjust this if you changed the normalization above.
    ])

    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


#can be customized
def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Implement your postprocessing steps here
    # For example, resizing back to original shape, converting to color mask, etc.
    # Return a numpy array suitable for saving as an image
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)  # Get the class with the highest probability
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # Remove batch and channel dimensions if necessary

    return prediction_numpy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set Transformers to offline mode to prevent the container from 
    # trying to reach the internet on the TU/e server
    import os
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Load model
    # Note: Model() uses your SegFormer architecture from model.py
    model = Model(n_classes=19) 
    
    # Load the weights
    # We use weights_only=False because SegFormer objects have complex structures
    state_dict = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=False, 
    )
    
    # Load into the model. We use strict=False because the HuggingFace wrapper 
    # often adds/renames internal metadata keys that don't affect prediction.
    model.load_state_dict(state_dict, strict=False)
    
    model.eval().to(device)
    print("Model loaded successfully.")





    image_files = list(Path(IMAGE_DIR).glob("*.png"))  # DO NOT CHANGE, IMAGES WILL BE PROVIDED IN THIS FORMAT
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass
            pred = model(img_tensor)

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Create mirrored output folder
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
