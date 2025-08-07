"""
Training script for LSDM (Landslide Detection Model)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import yaml
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import argparse
from tqdm import tqdm
import logging

from lsdm_model import create_lsdm_model


class LandslideDataset(Dataset):
    """
    Custom dataset for landslide detection
    Expects YOLO format annotations
    """
    
    def __init__(self, images_dir, labels_dir, img_size=640, augment=False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.img_size = img_size
        self.augment = augment
        
        # Get image files
        self.image_files = list(self.images_dir.glob("*.jpg")) + \
                          list(self.images_dir.glob("*.jpeg")) + \
                          list(self.images_dir.glob("*.png"))
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Load labels if available
        labels = torch.zeros((0, 5))  # Default empty labels
        if self.labels_dir:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    labels_list = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            labels_list.append([class_id, x_center, y_center, width, height])
                    
                    if labels_list:
                        labels = torch.tensor(labels_list, dtype=torch.float32)
        
        return image, labels


class YOLOLoss(nn.Module):
    """
    Simplified YOLO loss function
    """
    
    def __init__(self, num_classes=1):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.bce_obj = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        # Simplified loss calculation
        # In a complete implementation, this would include:
        # - Box regression loss (CIOU/DFL)
        # - Classification loss
        # - Objectness loss
        # - Proper target assignment
        
        total_loss = 0
        for pred in predictions:
            # Dummy loss for demonstration
            # Replace with proper YOLO loss implementation
            total_loss += pred.mean() * 0.1
            
        return total_loss


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'Loss': loss.item()})
        
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validation function"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            predictions = model(images)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_lsdm(args):
    """Main training function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = LandslideDataset(
        args.train_images, 
        args.train_labels, 
        args.img_size, 
        augment=True
    )
    
    val_dataset = LandslideDataset(
        args.val_images,
        args.val_labels,
        args.img_size,
        augment=False
    ) if args.val_images else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    ) if val_dataset else None
    
    # Create model
    model = create_lsdm_model(num_classes=args.num_classes).to(device)
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = YOLOLoss(num_classes=args.num_classes)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device) if val_loader else 0
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        print(f'Epoch {epoch}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        if val_loader:
            print(f'  Val Loss: {val_loss:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train LSDM model')
    
    # Data parameters
    parser.add_argument('--train-images', type=str, required=True, 
                       help='Path to training images directory')
    parser.add_argument('--train-labels', type=str, 
                       help='Path to training labels directory')
    parser.add_argument('--val-images', type=str,
                       help='Path to validation images directory')
    parser.add_argument('--val-labels', type=str,
                       help='Path to validation labels directory')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=1,
                       help='Number of classes (default: 1 for landslide)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./runs/train',
                       help='Output directory for models and logs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train_lsdm(args)


if __name__ == '__main__':
    main()