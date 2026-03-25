"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode
)

from model import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image

###################################################

#Metric class to compute IoU and Dice Coefficient for semantic segmentation
#only  computes the IoU and Dice for classes that actually appear in the targets to avoid skewing metrics with classes that are never present in the validation set (e.g., "train" or "motorcycle" in Cityscapes' val split)
# class SegmentationMetrics:
#     def __init__(self, num_classes=19, ignore_index=255):
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
#         self.total_intersections = torch.zeros(num_classes)
#         self.total_unions = torch.zeros(num_classes)
#         self.total_targets = torch.zeros(num_classes)
#         self.total_preds = torch.zeros(num_classes)

#     def update(self, preds, target):
#         preds = preds.contiguous().view(-1)
#         target = target.contiguous().view(-1)
        
#         mask = (target != self.ignore_index)
#         preds = preds[mask]
#         target = target[mask]
        
#         for cls in range(self.num_classes):
#             pred_inds = (preds == cls)
#             target_inds = (target == cls)
            
#             intersection = (pred_inds & target_inds).sum()
#             union = pred_inds.sum() + target_inds.sum() - intersection
            
#             self.total_intersections[cls] += intersection.cpu()
#             self.total_unions[cls] += union.cpu()
#             self.total_targets[cls] += target_inds.sum().cpu()
#             self.total_preds[cls] += pred_inds.sum().cpu()

#     def compute(self):
#         # Only compute for classes that actually appeared in the targets
#         valid_classes = self.total_targets > 0
        
#         ious = self.total_intersections[valid_classes] / torch.clamp(self.total_unions[valid_classes], min=1)
#         dices = (2.0 * self.total_intersections[valid_classes]) / torch.clamp(self.total_targets[valid_classes] + self.total_preds[valid_classes], min=1)
        
#         return ious.mean().item(), dices.mean().item()
###################################################



#---------------------New  SegmentationMetrics class with super-category metrics (uncomment if you want to use this instead of the simpler version above)---------------------# 
class SegmentationMetrics:
    def __init__(self, num_classes=19, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.total_intersections = torch.zeros(num_classes)
        self.total_unions = torch.zeros(num_classes)
        self.total_targets = torch.zeros(num_classes)
        self.total_preds = torch.zeros(num_classes)

    def update(self, preds, target):
        preds = preds.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        mask = (target != self.ignore_index)
        preds = preds[mask]
        target = target[mask]
        
        for cls in range(self.num_classes):
            pred_inds = (preds == cls)
            target_inds = (target == cls)
            
            intersection = (pred_inds & target_inds).sum()
            union = pred_inds.sum() + target_inds.sum() - intersection
            
            self.total_intersections[cls] += intersection.cpu()
            self.total_unions[cls] += union.cpu()
            self.total_targets[cls] += target_inds.sum().cpu()
            self.total_preds[cls] += pred_inds.sum().cpu()

    def _get_category_metrics(self, class_indices):
        """Helper to calculate IoU and Dice for a specific subset of classes"""
        valid = self.total_targets[class_indices] > 0
        if not valid.any():
            return 0.0, 0.0 # Return 0 if these classes aren't in the validation batch
        
        valid_indices = torch.tensor(class_indices)[valid]
        intersections = self.total_intersections[valid_indices]
        unions = self.total_unions[valid_indices]
        targets = self.total_targets[valid_indices]
        preds = self.total_preds[valid_indices]
        
        ious = intersections / torch.clamp(unions, min=1)
        dices = (2.0 * intersections) / torch.clamp(targets + preds, min=1)
        
        return ious.mean().item(), dices.mean().item()

    def compute(self):
        # Overall Mean Metrics
        mean_iou, mean_dice = self._get_category_metrics(list(range(self.num_classes)))
        
        # Super-Category: Human (11: Person, 12: Rider)
        human_iou, human_dice = self._get_category_metrics([11, 12])
        
        # Super-Category: Vehicle (13: Car, 14: Truck, 15: Bus, 16: Train, 17: Motorcycle, 18: Bicycle)
        vehicle_iou, vehicle_dice = self._get_category_metrics([13, 14, 15, 16, 17, 18])
        
        return mean_iou, mean_dice, human_iou, human_dice, vehicle_iou, vehicle_dice
#_---------------------New with different small categories (uncomment if you want to use these instead of the human/vehicle split)---------------------#



def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

# NEW: Flag to easily turn off W&B for local testing
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging")

    return parser


def main(args):
    # Initialize wandb for logging ONLY if not disabled
    if not args.disable_wandb:
        wandb.init(
            project="5lsm0-cityscapes-segmentation",
            name=args.experiment_id,
            config=vars(args),
        )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # #-------------------UNET-SPECIFIC TRANSFORMS-------------------#
    # # Define the transforms to apply to the data
    # img_transform = Compose([
    # ToImage(),
    # Resize((256, 256)),
    # ToDtype(torch.float32, scale=True),
    # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    # # Target transform (mask)
    # target_transform = Compose([
    #     ToImage(),
    #     Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    #     ToDtype(torch.int64),  # no scaling
    # ])
    # #-------------------UNET-SPECIFIC TRANSFORMS-------------------#

    #------------------------------------SEGMENTATION-SPECIFIC TRANSFORMS------------------------------------#
    # Define the transforms to apply to the data
    img_transform = Compose([
        ToImage(),
        Resize((256, 512)), # <--- CHANGED! ratio is now 1:2 to better match Cityscapes' original aspect ratio
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        Resize((256, 512), interpolation=InterpolationMode.NEAREST), # <--- CHANGED!
        ToDtype(torch.int64),
    ])
    #------------------------------------SEGMENTATION-SPECIFIC TRANSFORMS------------------------------------#

    # Load the dataset
    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define the model
    model = Model(in_channels=3, n_classes=19).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    #-------------------UNET OPTIMIZER-------------------#
    # Use a standard learning rate for the whole model
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #-------------------UNET OPTIMIZER-------------------#

    #-------------------SEGFORMER OPTIMIZER-------------------#
    # --- SEGFORMER OPTIMIZER (Commented out for now) ---
    # backbone_params = model.segformer.segformer.parameters()
    # head_params = model.segformer.decode_head.parameters()
    # optimizer = torch.optim.AdamW([
    #     {'params': backbone_params, 'lr': args.lr * 0.01}, 
    #     {'params': head_params, 'lr': args.lr}            
    # ])
    #-------------------SEGFORMER OPTIMIZER-------------------#

    #----------SegFormer-specific note on optimizers (uncomment if using SegFormer)----------#
    # 1. Use a standard optimizer for the whole model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #----------SegFormer-specific note on optimizers (uncomment if using SegFormer)----------#

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1:04}/{args.epochs:04} ---")

        # Training
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print to terminal every 10 batches so you can monitor progress in SLURM
            if (i + 1) % 10 == 0 or (i + 1) == len(train_dataloader):
                print(f"Batch [{i+1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}")

            # Universal W&B Logging (Works for U-Net, Base SegFormer, and Fine-Tuned SegFormer)
            if not args.disable_wandb:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[-1]['lr'], # Universal magic index!
                    "epoch": epoch + 1,
                }, step=epoch * len(train_dataloader) + i)

        # Validation
        model.eval()
        print("Running validation...")
        
        # Initialize the metric tracker for this epoch
        val_metrics = SegmentationMetrics(num_classes=19, ignore_index=255)
        
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                
                # Get the predicted classes (argmax over the channel dimension)
                predictions = outputs.softmax(1).argmax(1)
                
                # Update IoU and Dice metrics
                val_metrics.update(predictions, labels)
            
                # Only process and upload image grids if W&B is active!
                if i == 0 and not args.disable_wandb:
                    predictions_img_format = predictions.unsqueeze(1)
                    labels_img_format = labels.unsqueeze(1)

                    predictions_colored = convert_train_id_to_color(predictions_img_format)
                    labels_colored = convert_train_id_to_color(labels_img_format)

                    predictions_img = make_grid(predictions_colored.cpu(), nrow=8).permute(1, 2, 0).numpy()
                    labels_img = make_grid(labels_colored.cpu(), nrow=8).permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            # Calculate final validation metrics
            valid_loss = sum(losses) / len(losses)
            mean_iou, mean_dice, human_iou, human_dice, vehicle_iou, vehicle_dice = val_metrics.compute()
            
            print(f"Epoch {epoch+1} | Val Loss: {valid_loss:.4f} | Mean IoU: {mean_iou:.4f} | Human IoU: {human_iou:.4f} | Vehicle IoU: {vehicle_iou:.4f}")
            
            if not args.disable_wandb:
                wandb.log({
                    "valid_loss": valid_loss,
                    "mean_iou": mean_iou,
                    "mean_dice": mean_dice,
                    "human_iou": human_iou,
                    "human_dice": human_dice,
                    "vehicle_iou": vehicle_iou,
                    "vehicle_dice": vehicle_dice
                }, step=(epoch + 1) * len(train_dataloader) - 1)

            # Save best model (We can now save based on IoU instead of just loss!)
            if valid_loss < best_valid_loss: # You could change this to `if mean_iou > best_iou:` if you prefer!
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)
                print(f"New best model saved! (Loss: {valid_loss:.4f}, IoU: {mean_iou:.4f})")        
    print("\nTraining complete!")

    # Save the final model
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt")
    )
    
    if not args.disable_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)