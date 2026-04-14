"""
Enhanced Training Script: Multi-Source Deepfake Detection
Trains on FaceForensics++, Celeb-DF, and Synthetic Data

Usage:
  python train_multi_source.py --epochs 50 --batch-size 8 --lr 0.0001
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import SimpleMultimodalDetector
from training.multi_source_loader import create_multi_source_dataloaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multi_source_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiSourceTrainer:
    """Trainer for multi-source deepfake detection"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        logger.info(f"[OK] Using device: {self.device}")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Early stopping
        self.early_stopping_patience = args.early_stopping_patience
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': None,
            'test_acc': None,
            'best_epoch': None
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path('models/multi_source_checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (frames, labels) in enumerate(self.train_loader):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            
            # Handle tuple output
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                          f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in self.val_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                
                # Handle tuple output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def test(self):
        """Test on test set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for frames, labels in self.test_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                
                # Handle tuple output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, epochs):
        """Full training loop"""
        logger.info(f"\n{'='*60}")
        logger.info(f"MULTI-SOURCE TRAINING STARTED")
        logger.info(f"{'='*60}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch Size: {self.args.batch_size}")
        logger.info(f"Learning Rate: {self.args.lr}")
        logger.info(f"Device: {self.device}")
        logger.info(f"{'='*60}\n")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch [{epoch + 1}/{epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.history['best_epoch'] = epoch + 1
                
                checkpoint_path = self.checkpoint_dir / f'best_epoch_{epoch + 1}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"  [OK] Saved best model to {checkpoint_path}")
            else:
                self.early_stopping_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f'epoch_{epoch + 1}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"  [OK] Saved checkpoint to {checkpoint_path}")
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"\n[OK] Early stopping at epoch {epoch + 1}")
                break
        
        # Test on best model
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing on test set...")
        logger.info(f"{'='*60}")
        
        test_loss, test_acc, predictions, labels = self.test()
        self.history['test_loss'] = test_loss
        self.history['test_acc'] = test_acc
        
        logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        logger.info(f"{'='*60}\n")
        
        # Save final model
        final_model_path = 'models/pretrained/fusion_model_multi_source.pth'
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"[OK] Saved final model to {final_model_path}")
        
        # Save history
        history_path = 'reports/multi_source_training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            history_json = {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'train_acc': [float(x) for x in self.history['train_acc']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_acc': [float(x) for x in self.history['val_acc']],
                'test_loss': float(self.history['test_loss']),
                'test_acc': float(self.history['test_acc']),
                'best_epoch': self.history['best_epoch']
            }
            json.dump(history_json, f, indent=2)
        logger.info(f"[OK] Saved training history to {history_path}")
        
        # Plot results
        self._plot_results()
        
        return self.history
    
    def _plot_results(self):
        """Plot training results"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Loss
            axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
            axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy
            axes[1].plot(self.history['train_acc'], label='Train Accuracy', marker='o')
            axes[1].plot(self.history['val_acc'], label='Val Accuracy', marker='s')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = 'reports/multi_source_training_results.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"[OK] Saved plot to {plot_path}")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Source Deepfake Detection Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    # Create reports directory
    Path('reports').mkdir(exist_ok=True)
    
    # Load data
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading multi-source datasets...")
    logger.info(f"{'='*60}")
    
    dataloaders = create_multi_source_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        frames_per_video=30,
        num_workers=args.num_workers,
        include_all_sources=True
    )
    
    # Create model
    logger.info(f"\n{'='*60}")
    logger.info(f"Initializing model...")
    logger.info(f"{'='*60}")
    
    model = SimpleMultimodalDetector(num_classes=2, pretrained=True)
    
    # Create trainer
    trainer = MultiSourceTrainer(
        model,
        dataloaders['train'],
        dataloaders['val'],
        dataloaders['test'],
        args
    )
    
    # Train
    history = trainer.train(args.epochs)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"Best Epoch: {history['best_epoch']}")
    logger.info(f"Final Test Accuracy: {history['test_acc']:.2f}%")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
