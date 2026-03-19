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

    # Define the transforms to apply to the data
    img_transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

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

    # Define the optimizer with Differential Learning Rates
    backbone_params = model.segformer.segformer.parameters()
    head_params = model.segformer.decode_head.parameters()

    # Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.01}, 
        {'params': head_params, 'lr': args.lr}            
    ])

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

            if not args.disable_wandb:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate_head": optimizer.param_groups[1]['lr'],
                    "epoch": epoch + 1,
                }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        print("Running validation...")
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
                # Only process and upload image grids if W&B is active!
                if i == 0 and not args.disable_wandb:
                    predictions = outputs.softmax(1).argmax(1).unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8).permute(1, 2, 0).numpy()
                    labels_img = make_grid(labels.cpu(), nrow=8).permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            print(f"Epoch {epoch+1} Validation Loss: {valid_loss:.4f}")
            
            if not args.disable_wandb:
                wandb.log({"valid_loss": valid_loss}, step=(epoch + 1) * len(train_dataloader) - 1)

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)
                print(f"New best model saved! (Loss: {valid_loss:.4f})")
        
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