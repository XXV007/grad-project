"""
Simplified Multi-Source Deepfake Training
Works with FaceForensics, Celeb-DF, or either one
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import SimpleMultimodalDetector
from training.multi_source_loader import CelebDFDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/celebdf_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CelebDFTrainer:
    """Trainer for Celeb-DF deepfakes"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.early_stopping_patience = args.early_stopping_patience
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': None,
            'test_acc': None,
            'best_epoch': None
        }
        
        self.checkpoint_dir = Path('models/celebdf_checkpoints')
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
            
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % max(1, len(self.train_loader) // 3) == 0:
                logger.info(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in self.val_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
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
        """Test"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in self.test_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs):
        """Full training loop"""
        logger.info(f"\n{'='*60}")
        logger.info(f"CELEB-DF TRAINING")
        logger.info(f"{'='*60}")
        logger.info(f"Epochs: {epochs}, Batch Size: {self.args.batch_size}, LR: {self.args.lr}")
        logger.info(f"{'='*60}\n")
        
        for epoch in range(epochs):
            logger.info(f"Epoch [{epoch + 1}/{epochs}]")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.history['best_epoch'] = epoch + 1
                checkpoint_path = self.checkpoint_dir / f'best_epoch_{epoch + 1}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"  [OK] Saved best model")
            else:
                self.early_stopping_counter += 1
            
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f'epoch_{epoch + 1}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"[OK] Early stopping at epoch {epoch + 1}")
                break
        
        logger.info(f"\nTesting...")
        test_loss, test_acc = self.test()
        self.history['test_loss'] = test_loss
        self.history['test_acc'] = test_acc
        
        logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        final_model_path = 'models/pretrained/fusion_model_celebdf.pth'
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"[OK] Saved final model to {final_model_path}")
        
        history_path = 'reports/celebdf_training_history.json'
        with open(history_path, 'w') as f:
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
        logger.info(f"[OK] Saved training history")
        
        self._plot_results()
        
        return self.history
    
    def _plot_results(self):
        """Plot training results"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
            axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Celeb-DF Training - Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(self.history['train_acc'], label='Train Accuracy', marker='o')
            axes[1].plot(self.history['val_acc'], label='Val Accuracy', marker='s')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Celeb-DF Training - Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = 'reports/celebdf_training_results.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"[OK] Saved plot to {plot_path}")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Celeb-DF Deepfake Detection Training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data workers')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    
    Path('reports').mkdir(exist_ok=True)
    
    logger.info("Loading Celeb-DF dataset...")
    
    dataset_dir = 'data/celeb-deepfakeforensics-master'
    if not Path(dataset_dir).exists():
        logger.error(f"Dataset not found: {dataset_dir}")
        return
    
    train_dataset = CelebDFDataset(dataset_dir, split='train', frames_per_video=30)
    val_dataset = CelebDFDataset(dataset_dir, split='val', frames_per_video=30)
    test_dataset = CelebDFDataset(dataset_dir, split='test', frames_per_video=30)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = SimpleMultimodalDetector(num_classes=2, pretrained=True)
    
    trainer = CelebDFTrainer(model, train_loader, val_loader, test_loader, args)
    history = trainer.train(args.epochs)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"Best Epoch: {history['best_epoch']}")
    logger.info(f"Final Test Accuracy: {history['test_acc']:.2f}%")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
