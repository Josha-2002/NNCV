import os
import sys
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

# FORCE Python to check the /app directory so the Efficiency server doesn't crash
sys.path.append('/app')
from config import MODEL_SIZE, INIT_WEIGHTS  # Import the Central Brain!

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        
        if INIT_WEIGHTS == "cityscapes":
            # Automatically build the path for B0 or B1
            local_dir = f"/app/segformer-{MODEL_SIZE}-cityscapes"
            hf_path = f"nvidia/segformer-{MODEL_SIZE}-finetuned-cityscapes-1024-1024"
            pretrained_path = local_dir if os.path.exists(local_dir) else hf_path
            
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_path, num_labels=n_classes, ignore_mismatched_sizes=False
            )
            
            # FREEZE BACKBONE automatically
            for param in self.segformer.segformer.parameters():
                param.requires_grad = False
            print(f"Model initialized: Pre-trained Cityscapes {MODEL_SIZE.upper()}. Backbone FROZEN.")

        elif INIT_WEIGHTS == "blank":
            # Automatically build the path for mit-b0 or mit-b1
            local_dir = f"/app/mit-{MODEL_SIZE}"
            hf_path = f"nvidia/mit-{MODEL_SIZE}"
            pretrained_path = local_dir if os.path.exists(local_dir) else hf_path
            
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_path, num_labels=n_classes, ignore_mismatched_sizes=True
            )
            # DO NOT FREEZE BACKBONE
            print(f"Model initialized: Blank Slate (ImageNet) {MODEL_SIZE.upper()}. Backbone UNFROZEN.")
            
        else:
            raise ValueError("ERROR: INIT_WEIGHTS in config.py must be 'cityscapes' or 'blank'")

    def forward(self, x):
        outputs = self.segformer(pixel_values=x)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return upsampled_logits

