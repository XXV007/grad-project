"""
Data Loader Module
Handles dataset loading and preprocessing for training

CPSC 589 - Multimodal Deepfake Detection
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection
    """
    
    def __init__(self, data_dir, annotation_file, transform=None, 
                 sequence_length=30, mode='train'):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing video files or frames
            annotation_file: JSON file with labels
            transform: Image transforms
            sequence_length: Number of frames per sequence
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.mode = mode
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.samples = list(self.annotations.items())
        logger.info(f"Loaded {len(self.samples)} samples for {mode} mode")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample
        
        Returns:
            frames: Tensor of frames (sequence_length, C, H, W)
            label: 0 for real, 1 for fake
        """
        video_name, label = self.samples[idx]
        
        # Load frames for this video
        frames = self._load_frames(video_name)
        
        # Apply transforms
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        
        # Convert label
        label = 1 if label == 'fake' or label == 1 else 0
        
        return frames, label
    
    def _load_frames(self, video_name):
        """
        Load frames from video or frame directory
        
        Args:
            video_name: Name of video or frame directory
        
        Returns:
            frames: List of PIL Images
        """
        video_path = os.path.join(self.data_dir, video_name)
        
        # Check if it's a video file or directory of frames
        if os.path.isfile(video_path):
            return self._load_from_video(video_path)
        else:
            return self._load_from_frames(video_path)
    
    def _load_from_video(self, video_path):
        """Load frames from video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        target_frames = self.sequence_length
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        if total_frames > target_frames:
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        else:
            indices = list(range(total_frames))
        
        idx = 0
        while cap.isOpened() and len(frames) < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count in indices:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                idx += 1
            
            frame_count += 1
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < target_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
        
        return frames[:target_frames]
    
    def _load_from_frames(self, frame_dir):
        """Load frames from directory"""
        frame_files = sorted([f for f in os.listdir(frame_dir) 
                            if f.endswith(('.jpg', '.png'))])
        
        frames = []
        target_frames = self.sequence_length
        
        # Sample frames uniformly
        if len(frame_files) > target_frames:
            indices = np.linspace(0, len(frame_files) - 1, target_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        for frame_file in frame_files[:target_frames]:
            frame_path = os.path.join(frame_dir, frame_file)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
        
        # Pad if necessary
        while len(frames) < target_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
        
        return frames[:target_frames]


def create_data_loaders(config, train_transform, val_transform):
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration object
        train_transform: Transforms for training
        val_transform: Transforms for validation/test
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=os.path.join(config.DATA_FOLDER, 'train'),
        annotation_file=os.path.join(config.ANNOTATIONS_FOLDER, 'train.json'),
        transform=train_transform,
        sequence_length=config.SEQUENCE_LENGTH,
        mode='train'
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=os.path.join(config.DATA_FOLDER, 'val'),
        annotation_file=os.path.join(config.ANNOTATIONS_FOLDER, 'val.json'),
        transform=val_transform,
        sequence_length=config.SEQUENCE_LENGTH,
        mode='val'
    )
    
    test_dataset = DeepfakeDataset(
        data_dir=os.path.join(config.DATA_FOLDER, 'test'),
        annotation_file=os.path.join(config.ANNOTATIONS_FOLDER, 'test.json'),
        transform=val_transform,
        sequence_length=config.SEQUENCE_LENGTH,
        mode='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("Data Loader Module")
    print("This module provides dataset loading utilities for training")
    print("\nExample usage:")
    print("  dataset = DeepfakeDataset(data_dir, annotations, transform)")
    print("  loader = DataLoader(dataset, batch_size=16, shuffle=True)")
