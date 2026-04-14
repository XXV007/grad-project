# FaceForensics Integration Guide

## Overview

Your project now supports both synthetic training data and the FaceForensics++ dataset. This guide explains how to download, setup, and train with FaceForensics data.

## FaceForensics++ Dataset

**FaceForensics++** is a large-scale deepfake detection dataset consisting of:
- 1,000 original high-quality videos from YouTube
- Manipulated versions created with 4 methods:
  - **DeepFakes**: Deep learning-based face swapping
  - **Face2Face**: Real-time 3D face reconstruction
  - **FaceSwap**: Face-swapping algorithm
  - **NeuralTextures**: GAN-based face reenactment
  - **FaceShifter**: High-fidelity identity-preserving face swap

## Downloading FaceForensics++

Since FaceForensics++ is a large dataset, it requires registration:

1. **Request Access**:
   - Visit: https://github.com/ondyari/FaceForensics
   - Fill out the Google Form to request the download script
   - Wait for approval (typically 1-7 days)

2. **Download the Dataset**:
   - Once approved, you'll receive a download script
   - The script downloads videos in different compression levels:
     - `c0`: No compression (highest quality, largest)
     - `c23`: H.264 compression (standard)
     - `c40`: H.264 highest compression (fastest)

3. **Download Command Example**:
   ```bash
   python download.py --dataset FaceForensics --compression c23 --types all
   ```

## Dataset Structure

```
FaceForensics/
├── original/
│   ├── videos/
│   │   └── [original_videos].mp4
│   └── c23/
│       └── [compressed_originals].mp4
├── DeepFakes/
│   ├── c23/
│   │   └── [deepfake_videos].mp4
├── Face2Face/
│   ├── c23/
│   │   └── [face2face_videos].mp4
├── FaceSwap/
│   ├── c23/
│   │   └── [faceswap_videos].mp4
├── NeuralTextures/
│   ├── c23/
│   │   └── [neuraltextures_videos].mp4
└── FaceShifter/
    ├── c23/
    │   └── [faceshifter_videos].mp4
```

## Setup Instructions

### Option 1: Using Only Synthetic Data (Current)

By default, the project uses synthetic training data generated earlier:

```bash
# Train with synthetic data only
python training/train_with_faceforensics.py --epochs 20 --batch-size 4
```

### Option 2: To Include FaceForensics Data

After downloading FaceForensics++:

1. **Place dataset in project**:
   ```powershell
   # From PowerShell in your project directory
   Copy-Item -Path "path/to/FaceForensics" -Destination "data/FaceForensics" -Recurse
   ```

2. **Train with both synthetic and FaceForensics**:
   ```bash
   python training/train_with_faceforensics.py \
       --epochs 50 \
       --batch-size 8 \
       --faceforensics-dir data/FaceForensics \
       --synthetic-dir training/synthetic_data \
       --lr 0.0001
   ```

### Option 3: FaceForensics Only

```bash
python training/train_with_faceforensics.py \
    --epochs 50 \
    --batch-size 8 \
    --faceforensics-dir data/FaceForensics \
    --synthetic-dir "" \
    --lr 0.0001
   ```

## Training Configuration

The new training script (`train_with_faceforensics.py`) supports:

### Command Line Arguments:
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size per iteration (default: 4)
- `--lr`: Learning rate (default: 0.0001)
- `--faceforensics-dir`: Path to FaceForensics dataset
- `--synthetic-dir`: Path to synthetic data (default: training/synthetic_data)

### Hyperparameters (in training script):
- `NUM_FRAMES`: Frames per video (default: 10)
- `IMAGE_SIZE`: Input image size (default: 224x224)
- `SPATIAL_BACKBONE`: CNN backbone (default: efficientnet_b0)
- `TEMPORAL_TYPE`: Temporal module (default: lstm)
- `FUSION_TYPE`: Fusion method (default: concat)
- `EARLY_STOPPING_PATIENCE`: Patience for early stopping (default: 5)

## Data Loaders

The project includes two data loaders:

### 1. FaceForensicsDataset
- Loads individual FaceForensics manipulation methods
- Handles video decoding and frame extraction
- Supports different compression levels

### 2. HybridDataset
- Automatically loads both synthetic and FaceForensics data
- Intelligently samples frames from both sources
- Balanced binary classification (Real vs. Fake)

## Expected Results

### With Synthetic Data Only (Current):
- Training Accuracy: ~92%
- Validation Accuracy: ~65-70% (overfitting on synthetic)
- Training Time: ~5-10 minutes (CPU)

### With FaceForensics (Expected):
- Training Accuracy: ~95%+
- Validation Accuracy: ~80-90% (real deepfakes)
- Training Time: ~1-2 hours (CPU, depends on video count)

## Output Files

Training generates:

```
logs/
├── training_YYYYMMDD_HHMMSS.log     # Training log
└── training_history_YYYYMMDD.png    # Loss/accuracy plots

models/
├── pretrained/
│   ├── fusion_model_faceforensics.pth  # Final model
│   └── fusion_model.pth                 # Previous model
└── checkpoints/
    ├── model_best.pth                   # Best validation checkpoint
    └── model_epoch_X.pth                # Epoch checkpoints
```

## Important Notes

1. **License**: FaceForensics++ is released under specific terms of use
2. **Citation**: If you use the dataset, cite the original paper:
   ```
   @inproceedings{roessler2019faceforensicspp,
       author = {Röessler et al.},
       title = {FaceForensics++: Learning to Detect Manipulated Facial Images},
       booktitle = {ICCV 2019}
   }
   ```

3. **Performance**: The hybrid dataset with both synthetic and real data provides:
   - Better generalization
   - More robust deepfake detection
   - Real-world applicability

4. **File Formats**: Supports:
   - MP4 videos (.mp4)
   - JPG images (.jpg) from synthetic data
   - H.264 and other compression codecs

## Testing

To test the trained model:

```python
# Test on FaceForensics
python test_model.py --model models/pretrained/fusion_model_faceforensics.pth
```

## Troubleshooting

1. **ImportError for cv2**: Install OpenCV: `pip install opencv-python`
2. **Video loading errors**: Ensure FFmpeg is installed: `pip install av`
3. **Memory issues**: Reduce batch size or num_frames
4. **Slow training**: Use GPU acceleration (CUDA) if available

## Next Steps

1. ✓ Review and understand the FaceForensics structure
2. Download FaceForensics++ dataset (requires registration)
3. Place dataset in `data/FaceForensics/`
4. Run training: `python training/train_with_faceforensics.py --faceforensics-dir data/FaceForensics`
5. Monitor training in `logs/`
6. Evaluate on test set

## References

- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **Paper**: https://arxiv.org/abs/1901.08971
- **Project Website**: https://faceforensics.org/
