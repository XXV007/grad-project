"""
Enhanced training script with FaceForensics support
Trains on both synthetic and real deepfake detection data
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import MultimodalDetector
from training.faceforensics_loader import HybridDataset
from utils.metrics import compute_metrics


logger = logging.getLogger(__name__)


class TrainerWithFaceForensics:
    """Enhanced trainer supporting FaceForensics dataset"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Setup model
        self.model = MultimodalDetector(
            spatial_backbone=config.get('SPATIAL_BACKBONE', 'efficientnet_b0'),
            temporal_type=config.get('TEMPORAL_TYPE', 'lstm'),
            fusion_type=config.get('FUSION_TYPE', 'concat'),
            pretrained_backbone=config.get('USE_PRETRAINED_BACKBONE', False)
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('LEARNING_RATE', 0.0001),
            weight_decay=config.get('WEIGHT_DECAY', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('SCHEDULER_PATIENCE', 5),
            verbose=True
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.get('LOG_DIR', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def load_dataset(self):
        """Load hybrid dataset (synthetic + FaceForensics)"""
        synthetic_dir = self.config.get('SYNTHETIC_DATA_DIR', 'training/synthetic_data')
        faceforensics_dir = self.config.get('FACEFORENSICS_DIR')
        
        logger.info(f"Loading dataset from: {synthetic_dir}")
        if faceforensics_dir:
            logger.info(f"  with FaceForensics from: {faceforensics_dir}")
        
        # Load training dataset
        train_dataset = HybridDataset(
            synthetic_dir=synthetic_dir,
            faceforensics_dir=faceforensics_dir,
            num_frames=self.config.get('NUM_FRAMES', 10),
            image_size=self.config.get('IMAGE_SIZE', 224),
            train=True
        )
        
        # Load validation dataset
        val_dataset = HybridDataset(
            synthetic_dir=synthetic_dir,
            faceforensics_dir=faceforensics_dir,
            num_frames=self.config.get('NUM_FRAMES', 10),
            image_size=self.config.get('IMAGE_SIZE', 224),
            train=False
        )
        
        batch_size = self.config.get('BATCH_SIZE', 4)
        num_workers = self.config.get('NUM_WORKERS', 0)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            
            # Handle model output (may be tuple for spatial+temporal)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % self.config.get('LOG_INTERVAL', 10) == 0:
                acc = 100 * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs=10):
        """Complete training loop"""
        logger.info("Loading dataset...")
        train_loader, val_loader = self.load_dataset()
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                logger.info("Model improved, saving checkpoint")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.get('EARLY_STOPPING_PATIENCE', 5):
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Regular checkpoint
            if (epoch + 1) % self.config.get('CHECKPOINT_INTERVAL', 5) == 0:
                self.save_checkpoint(epoch)
        
        logger.info("Training completed")
        self.plot_training_history()
        self.save_final_model()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get('CHECKPOINT_DIR', 'models/checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = checkpoint_dir / 'model_best.pth'
        else:
            path = checkpoint_dir / f'model_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def save_final_model(self):
        """Save final trained model"""
        model_dir = Path(self.config.get('MODEL_FOLDER', 'models/pretrained'))
        model_dir.mkdir(parents=True, exist_ok=True)
        
        final_path = model_dir / 'fusion_model_faceforensics.pth'
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, final_path)
        logger.info(f"Final model saved: {final_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plot_dir = Path(self.config.get('LOG_DIR', 'logs'))
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Training history plot saved: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detector with FaceForensics')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--faceforensics-dir', type=str, default=None,
                       help='Path to FaceForensics dataset')
    parser.add_argument('--synthetic-dir', type=str, default='training/synthetic_data',
                       help='Path to synthetic data')
    
    args = parser.parse_args()
    
    # Config
    config = {
        'EPOCHS': args.epochs,
        'BATCH_SIZE': args.batch_size,
        'LEARNING_RATE': args.lr,
        'NUM_FRAMES': 10,
        'IMAGE_SIZE': 224,
        'SPATIAL_BACKBONE': 'efficientnet_b0',
        'TEMPORAL_TYPE': 'lstm',
        'FUSION_TYPE': 'concat',
        'USE_PRETRAINED_BACKBONE': False,
        'WEIGHT_DECAY': 1e-5,
        'SCHEDULER_PATIENCE': 5,
        'EARLY_STOPPING_PATIENCE': 5,
        'LOG_INTERVAL': 10,
        'CHECKPOINT_INTERVAL': 5,
        'LOG_DIR': 'logs',
        'CHECKPOINT_DIR': 'models/checkpoints',
        'MODEL_FOLDER': 'models/pretrained',
        'SYNTHETIC_DATA_DIR': args.synthetic_dir,
        'FACEFORENSICS_DIR': args.faceforensics_dir,
        'NUM_WORKERS': 0
    }
    
    trainer = TrainerWithFaceForensics(config)
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
