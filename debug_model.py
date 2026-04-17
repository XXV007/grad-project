"""
Debug script to diagnose model loading issues
"""
import os
import torch
import numpy as np
from config import config
from models.fusion_model import SimpleMultimodalDetector

# Setup
device = torch.device('cuda' if torch.cuda.is_available() and config['development'].USE_GPU else 'cpu')
print(f"Device: {device}")

# Initialize detector
print("\n=== Initializing Detector ===")
detector = SimpleMultimodalDetector(config['development'], device)

print(f"\nModel initialized: {detector.model}")
print(f"Has trained weights: {detector.has_trained_weights}")

# Check model state
print("\n=== Model Weight Statistics ===")
for name, param in detector.model.named_parameters():
    if 'weight' in name or 'bias' in name:
        mean = param.data.mean().item()
        std = param.data.std().item()
        print(f"{name}: mean={mean:.6f}, std={std:.6f}, shape={list(param.shape)}")
        if len(list(param.shape)) > 0 and list(param.shape)[0] <= 5:
            print(f"  Values: {param.data.flatten()[:5]}")

# Test with dummy input
print("\n=== Testing with Dummy Input ===")
dummy_frames = torch.randn(2, 30, 3, 224, 224).to(device)
print(f"Input shape: {dummy_frames.shape}")

try:
    with torch.no_grad():
        logits, spatial_feat, temporal_feat = detector.model(dummy_frames)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits values: {logits}")
    
    probabilities = torch.softmax(logits, dim=1)
    print(f"Probabilities: {probabilities}")
    
    confidence, prediction = torch.max(probabilities, dim=1)
    print(f"Predictions: {prediction.tolist()}")
    print(f"Confidence: {confidence.tolist()}")
    
except Exception as e:
    print(f"Error during forward pass: {e}")
    import traceback
    traceback.print_exc()

# Check checkpoint file
print("\n=== Checkpoint File Analysis ===")
checkpoint_path = config['development'].FUSION_MODEL_PATH
print(f"Checkpoint path: {checkpoint_path}")
print(f"Exists: {os.path.exists(checkpoint_path)}")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        if 'model_state_dict' in checkpoint:
            print(f"State dict keys: {list(checkpoint['model_state_dict'].keys())[:10]}")
        elif 'state_dict' in checkpoint:
            print(f"State dict keys: {list(checkpoint['state_dict'].keys())[:10]}")
        else:
            # Assume it's directly a state dict
            print(f"Direct state dict keys: {list(checkpoint.keys())[:10]}")
    
    print(f"Checkpoint size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
