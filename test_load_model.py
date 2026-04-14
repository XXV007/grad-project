#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test model loading"""

import torch
from models.fusion_model import MultimodalDetector, SimpleMultimodalDetector
from config import config

print("=" * 70)
print("TEST 1: Load checkpoint directly")
print("=" * 70)

checkpoint_path = "models/pretrained/fusion_model.pth"
print(f"\nLoading from: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('model_state_dict', checkpoint)
print(f"[OK] Checkpoint loaded: {len(state_dict)} keys")

print("\n" + "=" * 70)
print("TEST 2: Create model architecture")
print("=" * 70)

model = MultimodalDetector(
    spatial_backbone='efficientnet_b0',
    temporal_type='lstm',
    fusion_type='concat',
    pretrained_backbone=False
)
print(f"[OK] Model created: {len(dict(model.state_dict()))} keys")

print("\n" + "=" * 70)
print("TEST 3: Load state dict into model")
print("=" * 70)

try:
    model.load_state_dict(state_dict)
    print("[OK] State dict loaded successfully!")
except RuntimeError as e:
    print(f"[ERROR] {e}")
    print("\nTrying partial loading...")
    model.load_state_dict(state_dict, strict=False)
    print("[OK] Partial loading successful")

print("\n" + "=" * 70)
print("TEST 4: Test SimpleMultimodalDetector")
print("=" * 70)

cfg = config['development']
device = torch.device('cpu')

print(f"\nConfig FUSION_MODEL_PATH: {cfg.FUSION_MODEL_PATH}")
print(f"File exists: {__import__('os').path.exists(cfg.FUSION_MODEL_PATH)}")

try:
    detector = SimpleMultimodalDetector(cfg, device)
    print(f"[OK] Detector created")
    print(f"  has_trained_weights: {detector.has_trained_weights}")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("[COMPLETE] ALL TESTS DONE")
print("=" * 70)
