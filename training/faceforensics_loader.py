"""
FaceForensics++ Data Loader
Handles loading and preprocessing FaceForensics dataset
"""

import os
import json
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


class FaceForensicsDataset(Dataset):
    """
    Dataset loader for FaceForensics++ dataset
    
    Expected structure:
    dataset/
    ├── original/
    │   ├── videos/
    │   └── frames/
    ├── DeepFakes/
    │   └── c23/ (or c40/c0 for different compression levels)
    ├── FaceSwap/
    ├── Face2Face/
    ├── NeuralTextures/
    └── FaceShifter/
    """
    
    def __init__(self, root_dir, manipulation_type='all', compression='c23', 
                 num_frames=10, image_size=224, train=True, split_ratio=0.8):
        """
        Args:
            root_dir: Path to FaceForensics dataset root
            manipulation_type: 'all', 'DeepFakes', 'FaceSwap', 'Face2Face', 'NeuralTextures', 'FaceShifter'
            compression: 'c0' (no compression), 'c23' (H.264), 'c40' (highest)
            num_frames: Number of frames to extract per video
            image_size: Size to resize images to
            train: Whether to use training or validation split
            split_ratio: Train/val split ratio
        """
        self.root_dir = Path(root_dir)
        self.manipulation_type = manipulation_type
        self.compression = compression
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train
        self.split_ratio = split_ratio
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset samples"""
        # Get original (real) videos
        real_videos = self._get_videos('original')
        
        # Get manipulated videos
        fake_videos = []
        if self.manipulation_type == 'all':
            manipulation_types = ['DeepFakes', 'FaceSwap', 'Face2Face', 
                                'NeuralTextures', 'FaceShifter']
        else:
            manipulation_types = [self.manipulation_type]
        
        for manip_type in manipulation_types:
            fake_videos.extend(self._get_videos(manip_type))
        
        # Create samples with labels
        all_samples = []
        
        # Real videos (label=0)
        for video in real_videos:
            all_samples.append((video, 0))  # 0 = Real
        
        # Fake videos (label=1)
        for video in fake_videos:
            all_samples.append((video, 1))  # 1 = Fake
        
        # Split into train/val
        total = len(all_samples)
        split_idx = int(total * self.split_ratio)
        
        if self.train:
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        print(f"Loaded {len(self.samples)} samples ({'train' if self.train else 'val'})")
    
    def _get_videos(self, video_type):
        """Get list of video paths for a given type"""
        if video_type == 'original':
            pattern = str(self.root_dir / 'original' / '*' / f'{self.compression}' / '*.mp4')
        else:
            pattern = str(self.root_dir / video_type / f'{self.compression}' / '*.mp4')
        
        return glob.glob(pattern)
    
    def _extract_frames(self, video_path, num_frames):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            # If video has fewer frames than requested, use all frames
            frame_indices = list(range(total_frames))
        else:
            # Evenly space frame indices
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        # Pad with last frame if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        return frames[:num_frames]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        try:
            frames = self._extract_frames(video_path, self.num_frames)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return a dummy sample on error
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8) 
                     for _ in range(self.num_frames)]
        
        # Convert frames to tensors
        frame_tensors = []
        for frame in frames:
            img = Image.fromarray(frame)
            frame_tensors.append(self.transform(img))
        
        # Stack frames along time dimension
        frames_tensor = torch.stack(frame_tensors, dim=0)  # [T, C, H, W]
        
        return frames_tensor, label


class HybridDataset(Dataset):
    """
    Hybrid dataset that can load both synthetic and FaceForensics data
    """
    
    def __init__(self, synthetic_dir=None, faceforensics_dir=None, 
                 num_frames=10, image_size=224, train=True):
        self.samples = []
        
        # Load synthetic data if available
        if synthetic_dir and os.path.exists(synthetic_dir):
            self._load_synthetic_data(synthetic_dir)
        
        # Load FaceForensics data if available
        if faceforensics_dir and os.path.exists(faceforensics_dir):
            try:
                ff_dataset = FaceForensicsDataset(
                    faceforensics_dir,
                    train=train, 
                    num_frames=num_frames,
                    image_size=image_size
                )
                # Add FaceForensics samples
                self.samples.extend(ff_dataset.samples)
            except Exception as e:
                print(f"Could not load FaceForensics data: {e}")
        
        self.num_frames = num_frames
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Hybrid dataset: {len(self.samples)} total samples")
    
    def _load_synthetic_data(self, synthetic_dir):
        """Load synthetic training data"""
        annotations_file = os.path.join(synthetic_dir, 'annotations.json')
        frames_base = os.path.join(synthetic_dir, 'frames')
        
        if not os.path.exists(annotations_file):
            return
        
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        for video_name, label_data in annotations.items():
            video_frames_dir = os.path.join(frames_base, video_name)
            if os.path.exists(video_frames_dir):
                label = label_data.get('label', 0)
                self.samples.append((video_frames_dir, label, 'synthetic'))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if len(sample) == 3 and sample[2] == 'synthetic':
            # Synthetic data (frame directory)
            video_dir, label, _ = sample
            frame_files = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
            
            # Sample frames evenly
            if len(frame_files) > self.num_frames:
                indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
                frame_files = [frame_files[i] for i in indices]
            
            frames_tensor = []
            for frame_file in frame_files[:self.num_frames]:
                img = Image.open(frame_file).convert('RGB')
                frames_tensor.append(self.transform(img))
            
            while len(frames_tensor) < self.num_frames:
                frames_tensor.append(frames_tensor[-1])
            
            return torch.stack(frames_tensor, dim=0), label
        else:
            # FaceForensics data
            video_path, label = sample
            frames = self._extract_frames(video_path, self.num_frames)
            
            frame_tensors = []
            for frame in frames:
                img = Image.fromarray(frame)
                frame_tensors.append(self.transform(img))
            
            return torch.stack(frame_tensors, dim=0), label
    
    def _extract_frames(self, video_path, num_frames):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        return frames[:num_frames]
