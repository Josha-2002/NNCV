import os
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class Model(nn.Module):
    """ 
    SegFormer B0 architecture adapted for the Cityscapes CodaLab Challenge.
    Original paper: https://arxiv.org/abs/2105.15203
    """
    def __init__(self, 
                 in_channels=3, 
                 n_classes=19):
        super().__init__()
        
        # Check if the local folder exists (in the Docker container)
        # If it does, use it! If not (like on your PC), download it.
        pretrained_path = "/app/mit-b0" if os.path.exists("/app/mit-b0") else "nvidia/mit-b0"
        
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_path, 
            num_labels=n_classes,
            ignore_mismatched_sizes=True
        )
        # Differential Learning Rates
        # FREEZE THE BACKBONE to speed up training (Transfer Learning)
        # for param in self.segformer.segformer.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits
        
        # Upsample logits back to the original image size so loss functions work
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=x.shape[-2:],
            mode="bilinear", 
            align_corners=False
        )
        return upsampled_logits


