# Deepfake Detection Issue - Root Cause Analysis & Fix

## Problem Summary
The system returns "No deepfake detected" (predicts all videos as REAL) because the fusion model checkpoint is **incomplete and untrained**.

## Root Cause

### 1. **Missing Fusion Layer in Checkpoint**
The `fusion_model.pth` contains only:
- ✅ Backbone weights (EfficientNet-B0)
- ✅ Temporal model weights (LSTM)
- ❌ **Missing: Fusion layer (classification head)**

### 2. **Prediction Analysis**
With a dummy input test:
```
Logits: [[0.0312, -0.0451], [0.0322, -0.0470]]
Probabilities: [[0.5191, 0.4809], [0.5198, 0.4802]]
Prediction: [0, 0]  (REAL with only 51.9% confidence)
```

This indicates:
- Logits are near zero (untrained)
- Both classes have ~50% probability
- Argmax returns class 0 (REAL) because it's slightly higher
- The fusion layer is using **random initialization**

### 3. **Why All Videos Show "REAL"**
- Untrained logits always near 0
- Softmax produces ~0.5 probabilities for both classes  
- Argmax always picks class 0 (REAL) due to slight positive bias
- Result: Every video classified as REAL with ~51% confidence

## How the System Determines Real vs Fake

### Current Logic (In `app.py` line 238):
```python
'prediction': 'FAKE' if prediction == 1 else 'REAL',
```

### Model Architecture Decision Flow:
1. **Frame Extraction** → Extract 30 frames from video
2. **Spatial Analysis** → EfficientNet-B0 processes each frame
   - Output: 1280-dimensional feature vector per frame
3. **Temporal Analysis** → Bidirectional LSTM processes sequence
   - Input: 30 spatial features
   - Output: 512-dimensional temporal feature vector
4. **Fusion** → Concatenate spatial + temporal → 2D classification
   - Input: Combined feature (1280 + 512 = 1792 dims)
   - Output: 2 logits [logit_real, logit_fake]
5. **Classification**:
   - Apply softmax → probabilities
   - Argmax → prediction (0=REAL, 1=FAKE)
6. **Confidence** → probability of predicted class

### Expected Behavior (With Trained Weights):
- **Real videos**: Class 0 logit >> Class 1 logit → High confidence REAL
- **Fake videos**: Class 1 logit >> Class 0 logit → High confidence FAKE
- **Example**: Logits [5.2, -3.8] → 99.7% REAL confidence

## Solutions

### **Option 1: Quick Test (Recommended for immediate use)**
Create a properly trained checkpoint using dummy data or retrain:
```bash
python training/train_multi_source.py --epochs 1 --batch-size 1
```

### **Option 2: Initialize with Better Weights**
Modify the fusion layer initialization in `fusion_model.py`:
```python
# In MultimodalDetector.__init__, after creating fusion_layer:
with torch.no_grad():
    for layer in self.fusion_layer:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
```

### **Option 3: Use Pre-training + Fine-tuning**
- Train spatial model only on classification task
- Train temporal model only
- Then train fusion layer with both frozen (transfer learning)

## Testing the Fix

After applying any solution, verify with:
```bash
python debug_model.py
```

Expected output with trained model:
```
Logits: [[3.5, -2.1], [-2.0, 3.8]]  # Large differences
Probabilities: [[0.97, 0.03], [0.02, 0.98]]  # Clear predictions
Predictions: [0, 1]  # Mix of REAL and FAKE
Confidence: [0.97, 0.98]  # High confidence
```

## Verification

The detection system correctly determines real vs fake by:
1. ✅ Extracting spatial features (CNN learns manipulation artifacts)
2. ✅ Learning temporal patterns (LSTM detects motion inconsistencies)  
3. ✅ Fusing both streams (Combined decision)
4. ⚠️ **CURRENTLY BROKEN**: Using untrained fusion layer (random predictions)

**Once the model is trained, this will work correctly.**
