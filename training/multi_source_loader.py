"""
Multi-Source Deepfake Dataset Loader
Supports FaceForensics++, Celeb-DF, and Synthetic Data
"""

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import random
import logging

logger = logging.getLogger(__name__)

class CelebDFDataset(Dataset):
    """Load Celeb-DF dataset (590 real + 5639 fake videos)"""
    
    def __init__(self, root_dir, split='train', frames_per_video=30, image_size=224, 
                 split_ratio=(0.7, 0.15, 0.15)):
        """
        Args:
            root_dir: Path to Celeb-DF root directory
            split: 'train', 'val', or 'test'
            frames_per_video: Number of frames to sample
            image_size: Output image size
            split_ratio: (train, val, test) ratio
        """
        self.root_dir = Path(root_dir)
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        self.split = split
        
        # Expected structure:
        # Celeb-DF/
        #   Celeb-real/       (590 real videos)
        #   YouTube-real/     (300 real videos)
        #   Celeb-synthesis/  (5639 fake videos)
        
        self.videos = []
        self._load_dataset(split_ratio)
        
        logger.info(f"CelebDF {split}: Loaded {len(self.videos)} videos")
    
    def _load_dataset(self, split_ratio):
        """Load video list and split into train/val/test"""
        train_ratio, val_ratio, test_ratio = split_ratio
        
        # Load real videos
        real_dirs = []
        for real_dir in ['Celeb-real', 'YouTube-real']:
            real_path = self.root_dir / real_dir
            if real_path.exists():
                real_dirs.append(real_path)
        
        # Load fake videos
        fake_path = self.root_dir / 'Celeb-synthesis'
        
        real_videos = []
        fake_videos = []
        
        # Collect real videos
        for real_dir in real_dirs:
            if real_dir.exists():
                for video_file in real_dir.glob('*.mp4'):
                    real_videos.append((str(video_file), 0))  # Label 0 = real
        
        # Collect fake videos
        if fake_path.exists():
            for video_file in fake_path.glob('*.mp4'):
                fake_videos.append((str(video_file), 1))  # Label 1 = fake
        
        all_videos = real_videos + fake_videos
        random.shuffle(all_videos)
        
        # Split into train/val/test
        n_train = int(len(all_videos) * train_ratio)
        n_val = int(len(all_videos) * val_ratio)
        
        if self.split == 'train':
            self.videos = all_videos[:n_train]
        elif self.split == 'val':
            self.videos = all_videos[n_train:n_train + n_val]
        else:  # test
            self.videos = all_videos[n_train + n_val:]
    
    def _extract_frames(self, video_path, num_frames):
        """Extract evenly-spaced frames from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < num_frames:
                # If video has fewer frames than needed, duplicate last frame
                indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
            else:
                # Sample evenly-spaced frames
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) < num_frames:
                # Pad with last frame
                frames.extend([frames[-1]] * (num_frames - len(frames)))
            
            return np.array(frames[:num_frames]), True
        except Exception as e:
            logger.warning(f"Error processing {video_path}: {e}")
            return None, False
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        frames, success = self._extract_frames(video_path, self.frames_per_video)
        
        if not success or frames is None:
            # Return dummy data on error
            frames = np.zeros((self.frames_per_video, self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
        frames = (frames - mean) / std
        
        # Convert to (T, C, H, W) format
        frames = torch.from_numpy(frames.transpose(0, 3, 1, 2).copy()).float()
        
        return frames, torch.tensor(label, dtype=torch.long)


class TripleFusionDataset(Dataset):
    """
    Fuse FaceForensics, Celeb-DF, and Synthetic data into one mega-dataset
    Supports three distinct deepfake sources for robust training
    """
    
    def __init__(self, root_dir, split='train', frames_per_video=30, image_size=224,
                 include_faceforensics=True, include_celebdf=True, include_synthetic=True):
        """
        Args:
            root_dir: Parent directory containing all datasets
            split: 'train', 'val', or 'test'
            frames_per_video: Frames to extract from each video
            image_size: Output image dimensions
            include_*: Whether to include each data source
        """
        self.root_dir = Path(root_dir)
        self.datasets = []
        
        # FaceForensics
        if include_faceforensics:
            ff_path = self.root_dir / 'FaceForensics'
            if ff_path.exists():
                logger.info(f"Loading FaceForensics from {ff_path}")
                from faceforensics_loader import HybridDataset
                is_train = (split == 'train')
                ff_dataset = HybridDataset(
                    faceforensics_dir=str(ff_path),
                    synthetic_dir=None,
                    num_frames=frames_per_video,
                    image_size=image_size,
                    train=is_train
                )
                if len(ff_dataset) > 0:
                    self.datasets.append(('FaceForensics', ff_dataset))
                    logger.info(f"  [OK] FaceForensics loaded: {len(ff_dataset)} samples")
        
        # Celeb-DF
        if include_celebdf:
            cdf_path = self.root_dir / 'celeb-deepfakeforensics-master'
            if cdf_path.exists():
                logger.info(f"Loading Celeb-DF from {cdf_path}")
                cdf_dataset = CelebDFDataset(
                    root_dir=str(cdf_path),
                    split=split,
                    frames_per_video=frames_per_video,
                    image_size=image_size
                )
                if len(cdf_dataset) > 0:
                    self.datasets.append(('CelebDF', cdf_dataset))
                    logger.info(f"  [OK] Celeb-DF loaded: {len(cdf_dataset)} samples")
        
        # Synthetic
        if include_synthetic:
            syn_path = self.root_dir / 'synthetic'
            if syn_path.exists():
                logger.info(f"Loading Synthetic data from {syn_path}")
                try:
                    from faceforensics_loader import HybridDataset
                    is_train = (split == 'train')
                    syn_dataset = HybridDataset(
                        faceforensics_dir=None,
                        synthetic_dir=str(syn_path),
                        num_frames=frames_per_video,
                        image_size=image_size,
                        train=is_train
                    )
                    if len(syn_dataset) > 0:
                        self.datasets.append(('Synthetic', syn_dataset))
                        logger.info(f"  [OK] Synthetic loaded: {len(syn_dataset)} samples")
                except Exception as e:
                    logger.warning(f"Could not load synthetic dataset: {e}")
        
        if not self.datasets:
            raise ValueError("No datasets found! Check paths and settings.")
        
        # Concatenate all datasets
        all_datasets = [ds for _, ds in self.datasets]
        self.combined_dataset = ConcatDataset(all_datasets)
        
        # Log summary
        total_samples = len(self.combined_dataset)
        logger.info(f"\n{'='*60}")
        logger.info(f"TRIPLE-FUSION DATASET LOADED ({split})")
        logger.info(f"{'='*60}")
        for name, ds in self.datasets:
            pct = 100.0 * len(ds) / total_samples
            logger.info(f"  {name:20s}: {len(ds):6d} samples ({pct:5.1f}%)")
        logger.info(f"{'='*60}")
        logger.info(f"  TOTAL              : {total_samples:6d} samples")
        logger.info(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        return self.combined_dataset[idx]
    
    def get_dataset_distribution(self):
        """Returns dictionary of dataset sizes"""
        return {name: len(ds) for name, ds in self.datasets}


def create_multi_source_dataloaders(data_dir, batch_size=4, frames_per_video=30,
                                    num_workers=4, include_all_sources=True):
    """
    Create train/val/test dataloaders from multiple sources
    
    Args:
        data_dir: Root directory containing all datasets
        batch_size: Batch size
        frames_per_video: Frames per video
        num_workers: Number of data loading workers
        include_all_sources: Use all available datasets
    
    Returns:
        Dict with 'train', 'val', 'test' dataloaders
    """
    
    train_dataset = TripleFusionDataset(
        data_dir, split='train', frames_per_video=frames_per_video,
        include_faceforensics=include_all_sources,
        include_celebdf=include_all_sources,
        include_synthetic=include_all_sources
    )
    
    val_dataset = TripleFusionDataset(
        data_dir, split='val', frames_per_video=frames_per_video,
        include_faceforensics=include_all_sources,
        include_celebdf=include_all_sources,
        include_synthetic=False  # Use less synthetic for validation
    )
    
    test_dataset = TripleFusionDataset(
        data_dir, split='test', frames_per_video=frames_per_video,
        include_faceforensics=include_all_sources,
        include_celebdf=include_all_sources,
        include_synthetic=False  # No synthetic for testing
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Test the multi-source loader
    logging.basicConfig(level=logging.INFO)
    
    data_dir = 'data'
    loaders = create_multi_source_dataloaders(data_dir, batch_size=4)
    
    print("\nTrain loader test:")
    for batch_idx, (frames, labels) in enumerate(loaders['train']):
        print(f"  Batch {batch_idx}: frames shape {frames.shape}, labels {labels}")
        if batch_idx >= 2:
            break
