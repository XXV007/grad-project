#!/usr/bin/env python3
"""
Diagnostic script to understand dataset structure
"""

import os
from pathlib import Path
import sys

def analyze_celebdf():
    """Analyze Celeb-DF structure"""
    print("\n" + "="*60)
    print("ANALYZING CELEB-DF STRUCTURE")
    print("="*60)
    
    base_path = Path("data/celeb-deepfakeforensics-master")
    if not base_path.exists():
        print("[ERROR] Celeb-DF not found at: data/celeb-deepfakeforensics-master")
        return False
    
    print(f"[OK] Found Celeb-DF at: {base_path}")
    
    # Check subdirectories
    subdirs = list(base_path.glob("*/"))
    print(f"[OK] Found {len(subdirs)} subdirectories:")
    for subdir in subdirs[:10]:
        name = subdir.name
        # Count files
        mp4_count = len(list(subdir.glob("*.mp4")))
        av_count = len(list(subdir.glob("*.avi")))
        mkv_count = len(list(subdir.glob("*.mkv")))
        print(f"     {name:30s} - {mp4_count:4d} MP4, {av_count:4d} AVI, {mkv_count:4d} MKV")
    
    return True

def analyze_faceforensics():
    """Analyze FaceForensics structure"""
    print("\n" + "="*60)
    print("ANALYZING FACEFORENSICS STRUCTURE")
    print("="*60)
    
    base_path = Path("data/FaceForensics")
    if not base_path.exists():
        print("[ERROR] FaceForensics not found at: data/FaceForensics")
        return False
    
    print(f"[OK] Found FaceForensics at: {base_path}")
    
    # Check subdirectories (manipulation types)
    subdirs = list(base_path.glob("*/"))
    print(f"[OK] Found {len(subdirs)} manipulation type directories:")
    for subdir in sorted(subdirs)[:10]:
        name = subdir.name
        # Count video files
        video_count = len(list(subdir.glob("**/*.mp4")))
        mask_count = len(list(subdir.glob("**/*.npy")))
        print(f"     {name:30s} - {video_count:6d} videos, {mask_count:6d} masks")
    
    return True

def analyze_synthetic():
    """Analyze Synthetic data structure"""
    print("\n" + "="*60)
    print("ANALYZING SYNTHETIC DATA STRUCTURE")
    print("="*60)
    
    base_path = Path("data/synthetic")
    if not base_path.exists():
        print("[ERROR] Synthetic data not found at: data/synthetic")
        return False
    
    print(f"[OK] Found Synthetic data at: {base_path}")
    
    # Check subdirectories
    subdirs = list(base_path.glob("*/"))
    print(f"[OK] Found {len(subdirs)} subdirectories:")
    for subdir in sorted(subdirs):
        name = subdir.name
        file_count = len(list(subdir.glob("**/*")))
        print(f"     {name:30s} - {file_count:6d} files")
    
    return True

if __name__ == "__main__":
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "DATASET STRUCTURE ANALYSIS" + " "*17 + "║")
    print("╚" + "═"*58 + "╝")
    
    celebdf_ok = analyze_celebdf()
    faceforensics_ok = analyze_faceforensics()
    synthetic_ok = analyze_synthetic()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Celeb-DF:       {'[OK] Available' if celebdf_ok else '[ERROR] Not Found'}")
    print(f"FaceForensics:  {'[OK] Available' if faceforensics_ok else '[ERROR] Not Found'}")
    print(f"Synthetic:      {'[OK] Available' if synthetic_ok else '[ERROR] Not Found'}")
    print("="*60 + "\n")
