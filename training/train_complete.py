"""
Complete Training Script for Spatial Model
Trains the spatial feature extractor on real or synthetic data

CPSC 589 - Multimodal Deepfake Detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import sys
import json
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.spatial_model import SpatialFeatureExtractor
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameDataset(Dataset):
    """Dataset that loads frames from disk"""
    
    def __init__(self, data_dir, annotation_file, transform=None, num_frames=10):
        """
        Args:
            data_dir: Directory containing video folders with frames
            annotation_file: JSON file with video_name -> label mappings
            transform: Image transformations
            num_frames: Number of frames to sample from each video
        """
        self.data_dir = data_dir
        self.transform = transform
        self.num_frames = num_frames
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.samples = list(self.annotations.items())
        logger.info(f"Loaded {len(self.samples)} samples from {annotation_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Single representative frame from the video (C, H, W)
            label: 0 for real, 1 for fake
        """
        video_name, label_str = self.samples[idx]
        
        # Get video directory
        video_dir = os.path.join(self.data_dir, video_name)
        
        if not os.path.exists(video_dir):
            logger.warning(f"Video directory not found: {video_dir}")
            # Return dummy image
            dummy = torch.zeros(3, 224, 224)
            return dummy, int(label_str == "fake" or label_str == 1)
        
        # Get all frames
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        
        if not frame_files:
            logger.warning(f"No frames found in {video_dir}")
            dummy = torch.zeros(3, 224, 224)
            return dummy, int(label_str == "fake" or label_str == 1)
        
        # Sample frames uniformly
        indices = np.linspace(0, len(frame_files)-1, self.num_frames, dtype=int)
        sampled_frames = [frame_files[i] for i in indices]
        
        # Load and average frames
        images = []
        for frame_file in sampled_frames:
            frame_path = os.path.join(video_dir, frame_file)
            try:
                img = Image.open(frame_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                logger.warning(f"Could not load {frame_path}: {e}")
                continue
        
        if not images:
            dummy = torch.zeros(3, 224, 224)
            return dummy, int(label_str == "fake" or label_str == 1)
        
        # Stack and average frames
        stacked = torch.stack(images)  # (num_frames, 3, 224, 224)
        averaged = stacked.mean(dim=0)  # (3, 224, 224)
        
        # Convert label
        label = int(label_str == "fake" or label_str == 1)
        
        return averaged, label


def create_dataloaders(data_dir, annotation_dir, batch_size=16, num_workers=0):
    """Create train, val, test dataloaders"""
    
    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        annotation_file = os.path.join(annotation_dir, f'{split}_annotations.json')
        
        if not os.path.exists(annotation_file):
            logger.warning(f"Annotation file not found: {annotation_file}")
            continue
        
        transform = train_transform if split == 'train' else val_transform
        
        dataset = FrameDataset(
            data_dir=data_dir,
            annotation_file=annotation_file,
            transform=transform,
            num_frames=10
        )
        datasets[split] = dataset
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers
        )
    
    return dataloaders, datasets


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle tuple output (logits, features)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # Handle tuple output (logits, features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_spatial_model(args):
    """Main training function"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders, datasets = create_dataloaders(
        data_dir=args.data_dir,
        annotation_dir=args.annotation_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if 'train' not in dataloaders:
        logger.error("Could not create training dataloader!")
        return
    
    # Initialize model
    logger.info(f"Initializing model: {args.backbone}")
    model = SpatialFeatureExtractor(
        backbone=args.backbone,
        pretrained=args.pretrained_backbone,
        num_classes=2
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    logger.info("=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Training samples: {len(datasets['train'])}")
    if 'val' in datasets:
        logger.info(f"Validation samples: {len(datasets['val'])}")
    logger.info("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Training
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        logger.info(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validation
        if 'val' in dataloaders:
            val_loss, val_acc = validate(model, dataloaders['val'], criterion, device)
            logger.info(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_path = os.path.join(args.checkpoint_dir, 'spatial_model_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, best_path)
                logger.info(f"  ✓ Saved best model: {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'spatial_model_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'spatial_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_path)
    logger.info(f"\n✓ Final model saved: {final_path}")
    
    # Also save as fusion_model.pth for compatibility with inference
    fusion_path = os.path.join(args.checkpoint_dir, 'fusion_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, fusion_path)
    logger.info(f"✓ Fusion model saved: {fusion_path}")
    
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Spatial Model for Deepfake Detection')
    
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'efficientnet_b4', 'xception', 'resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--data-dir', type=str, default='./data/raw',
                       help='Directory containing video frames')
    parser.add_argument('--annotation-dir', type=str, default='./data/annotations',
                       help='Directory containing annotation files')
    parser.add_argument('--checkpoint-dir', type=str, default='./models/pretrained',
                       help='Directory to save checkpoints')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of dataloader workers')
    parser.add_argument('--pretrained-backbone', action='store_true',
                       help='Use pretrained backbone weights')
    
    args = parser.parse_args()
    
    try:
        train_spatial_model(args)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)
