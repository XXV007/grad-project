"""
Synthetic Training Data Generator
Creates dummy video frames and annotations for training demonstrations

CPSC 589 - Multimodal Deepfake Detection
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_frames(output_dir, num_real=50, num_fake=50, frames_per_video=30):
    """
    Generate synthetic frame sequences for training
    
    Args:
        output_dir: Directory to save frames
        num_real: Number of real video sequences to generate
        num_fake: Number of fake video sequences to generate
        frames_per_video: Number of frames per sequence
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frame_size = (224, 224)
    annotations = {}
    
    # Generate real videos
    logger.info(f"Generating {num_real} real video sequences...")
    for vid_idx in range(num_real):
        video_name = f"real_video_{vid_idx:04d}"
        video_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)
        
        # Generate frames with natural variation
        for frame_idx in range(frames_per_video):
            # Create a frame with face-like pattern (blurred circular region)
            frame = np.ones((*frame_size, 3), dtype=np.uint8) * 200
            
            # Add some variation
            noise = np.random.randn(*frame_size, 3) * 20
            frame = np.clip(frame + noise.astype(np.uint8), 0, 255)
            
            # Add a "face" region (circular gradient)
            cy, cx = frame_size[0]//2, frame_size[1]//2
            y, x = np.ogrid[:frame_size[0], :frame_size[1]]
            mask = (x - cx)**2 + (y - cy)**2 <= (frame_size[0]//3)**2
            frame[mask] = np.clip(frame[mask] * 0.8, 0, 255)
            
            frame_path = os.path.join(video_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        annotations[video_name] = "real"
        if (vid_idx + 1) % 10 == 0:
            logger.info(f"  Generated {vid_idx + 1}/{num_real} real sequences")
    
    # Generate fake videos (with more artifacts/variations)
    logger.info(f"Generating {num_fake} fake video sequences...")
    for vid_idx in range(num_fake):
        video_name = f"fake_video_{vid_idx:04d}"
        video_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)
        
        # Generate frames with unnatural variation (simulating glitches/artifacts)
        for frame_idx in range(frames_per_video):
            frame = np.ones((*frame_size, 3), dtype=np.uint8) * 180
            
            # Add higher frequency noise (simulating compression artifacts)
            noise = np.random.randn(*frame_size, 3) * 40
            frame = np.clip(frame + noise.astype(np.uint8), 0, 255)
            
            # Add color shifts (simulating deepfake artifacts)
            if np.random.rand() > 0.5:
                frame[:, :, 0] = np.clip(frame[:, :, 0] * 0.9, 0, 255)  # Reduce red
            if np.random.rand() > 0.5:
                frame[:, :, 1] = np.clip(frame[:, :, 1] * 1.1, 0, 255)  # Increase green
            
            # Add "face" region with distortion
            cy, cx = frame_size[0]//2 + np.random.randint(-10, 10), frame_size[1]//2 + np.random.randint(-10, 10)
            y, x = np.ogrid[:frame_size[0], :frame_size[1]]
            mask = (x - cx)**2 + (y - cy)**2 <= (frame_size[0]//3)**2
            frame[mask] = np.clip(frame[mask] * 0.6, 0, 255)  # More distortion
            
            frame_path = os.path.join(video_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        annotations[video_name] = "fake"
        if (vid_idx + 1) % 10 == 0:
            logger.info(f"  Generated {vid_idx + 1}/{num_fake} fake sequences")
    
    return annotations


def create_split_annotations(annotations, output_dir, train_split=0.7, val_split=0.15):
    """
    Split annotations into train/val/test sets
    
    Args:
        annotations: Full annotation dictionary
        output_dir: Directory to save annotation files
        train_split: Proportion for training
        val_split: Proportion for validation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    samples = list(annotations.items())
    np.random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_data = dict(samples[:n_train])
    val_data = dict(samples[n_train:n_train+n_val])
    test_data = dict(samples[n_train+n_val:])
    
    # Save annotation files
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        annotation_path = os.path.join(output_dir, f'{split_name}_annotations.json')
        with open(annotation_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved {split_name} annotations: {annotation_path} ({len(split_data)} samples)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Synthetic Training Data')
    parser.add_argument('--num-real', type=int, default=50, help='Number of real video sequences')
    parser.add_argument('--num-fake', type=int, default=50, help='Number of fake video sequences')
    parser.add_argument('--frames-per-video', type=int, default=30, help='Frames per sequence')
    parser.add_argument('--output-dir', type=str, default='./data/raw')
    parser.add_argument('--annotations-dir', type=str, default='./data/annotations')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Generating Synthetic Training Data")
    logger.info("=" * 60)
    logger.info(f"Real sequences: {args.num_real}")
    logger.info(f"Fake sequences: {args.num_fake}")
    logger.info(f"Frames per sequence: {args.frames_per_video}")
    
    # Generate frames
    annotations = generate_synthetic_frames(
        args.output_dir,
        num_real=args.num_real,
        num_fake=args.num_fake,
        frames_per_video=args.frames_per_video
    )
    
    # Create splits
    create_split_annotations(annotations, args.annotations_dir)
    
    logger.info("=" * 60)
    logger.info("✓ Synthetic data generation complete!")
    logger.info(f"  Frames: {args.output_dir}")
    logger.info(f"  Annotations: {args.annotations_dir}")
    logger.info("=" * 60)
