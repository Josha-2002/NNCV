"""
AUTONOMOUS DRIVING with SEMANTIC SEGMENTATION - CENTRAL CONFIGURATION

This file serves as the Configuration for the pipeline. Modifying these 
variables dynamically updates `model.py`, `train.py`, and `predict.py` without requiring 
manual code changes.

All models are evaluated using standard FP32 (32-bit floating point) precision.

--- OFFICIAL PAPER & COMPETITION SUBMISSION CONFIGURATIONS ---

1. EFFICIENCY TRACK ("JK_HIGHWAY_SF_B0_Adaptive")
   - Objective: Maximize mathematical efficiency (MeanDice / TFLOPs) and inference speed.
   - Configuration:
        HEIGHT = 384
        WIDTH = 768
        MODEL_SIZE = "b0"
        INIT_WEIGHTS = "cityscapes"
   - Rationale: Using the compressed 384x768 resolution drastically reduces 
     the pixel count, utilizing the 3.7M parameter B0 model.

2. PEAK PERFORMANCE & PAPER FOCUS ("JK_URBAN_SF_B1_Adaptive")
   - Objective: Maximize overall accuracy and minority class safety (Humans/Vehicles) 
     while strictly maintaining Urban Autonomous Driving constraints (>= 30 FPS).
   - Configuration:
        HEIGHT = 384
        WIDTH = 768
        MODEL_SIZE = "b1"
        INIT_WEIGHTS = "cityscapes"
   - Rationale: The B1 architecture (13.7M parameters) provides spatial 
     understanding. At 384x768 FP32, inference operates at ~42.5 FPS, safely clearing 
     the 30 FPS real-time urban threshold. For the paper's custom safety model, the 
     backbone was frozen to prevent gradient shock, and the head was fine-tuned using 
     sharp class weights (3.0x for humans).
"""

# --- 1. RESOLUTION SETTINGS ---
# Set to (384, 768) for Efficiency Track submissions
# Set to (384, 768) for Peak Performance / Real-time Urban testing
HEIGHT = 384   
WIDTH = 768     
IMG_SIZE = (HEIGHT, WIDTH)

# --- 2. MODEL ARCHITECTURE ---
# Options: "b0" (3.7M params) or "b1" (13.7M params)
MODEL_SIZE = "b1" 

# --- 3. TRAINING STRATEGY ---
# Options: 
# "cityscapes" -> Loads fully trained model. Freezes backbone. Uses filtered optimizer.
# "blank"      -> Loads ImageNet pre-trained (mit). Unfreezes backbone. Uses differential optimizer.
INIT_WEIGHTS = "blank" 

print(f"Config Loaded: SegFormer-{MODEL_SIZE.upper()} (FP32) | Weights: {INIT_WEIGHTS} | Size: {IMG_SIZE}")