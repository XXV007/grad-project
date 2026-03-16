# Multimodal Deepfake Detection - Project Documentation

## System Architecture

### Overview
The system implements a two-stream architecture that processes videos through parallel spatial and temporal analysis pipelines before fusing the features for final classification.

### Components

#### 1. Spatial Feature Extractor (`models/spatial_model.py`)
- **Architecture**: EfficientNet-B0/B4, XceptionNet, or ResNet50
- **Purpose**: Extracts frame-level visual features
- **Input**: Individual frames (224x224x3)
- **Output**: Feature vectors (1280-2048 dimensions)
- **Detection Targets**: Pixel-level artifacts, texture inconsistencies, blending errors

#### 2. Temporal Analyzer (`models/temporal_model.py`)
- **Architecture**: LSTM (Bidirectional) or Temporal Transformer
- **Purpose**: Captures motion dynamics and temporal inconsistencies
- **Input**: Sequence of spatial features (30 frames)
- **Output**: Temporal feature vector (1024 dimensions for LSTM)
- **Detection Targets**: Unnatural blinking, irregular head movements, lip-sync issues

#### 3. Fusion Model (`models/fusion_model.py`)
- **Strategy**: Late fusion with concatenation or attention
- **Purpose**: Combines spatial and temporal evidence
- **Input**: Spatial features + Temporal features
- **Output**: Binary classification (Real/Fake) with confidence score

#### 4. Video Preprocessor (`utils/preprocessing.py`)
- Face detection using MediaPipe or Haar Cascades
- Frame extraction at configurable FPS
- Face alignment and cropping
- Normalization and augmentation

#### 5. Explainability Module (`utils/explainability.py`)
- Grad-CAM for spatial attention visualization
- Temporal activation heatmaps
- Frame-by-frame anomaly scores

### Data Flow

```
Input Video
    ↓
Frame Extraction (10 FPS)
    ↓
Face Detection & Cropping
    ↓
Preprocessing & Normalization
    ↓
Spatial CNN (per frame) → Spatial Features
    ↓
Temporal Model (sequence) → Temporal Features
    ↓
Feature Fusion
    ↓
Classification (Real/Fake + Confidence)
    ↓
Explainability Visualizations
```

## Training Pipeline

### Dataset Preparation
1. Download datasets (FaceForensics++, DFDC, Celeb-DF)
2. Extract frames and faces
3. Split into train/val/test sets
4. Apply augmentation (compression, noise, blur)

### Training Stages

#### Stage 1: Train Spatial Model
```bash
python training/train_spatial.py --backbone efficientnet_b4 --epochs 50 --batch_size 32
```

#### Stage 2: Train Temporal Model
```bash
python training/train_temporal.py --model_type lstm --epochs 50 --batch_size 16
```

#### Stage 3: Train Fusion Model
```bash
python training/train_fusion.py --fusion_type concat --epochs 30 --batch_size 8
```

### Hyperparameters
- Learning Rate: 1e-4 (with ReduceLROnPlateau)
- Optimizer: Adam with weight decay (1e-5)
- Batch Size: 16-32 (depending on GPU memory)
- Sequence Length: 30 frames
- Early Stopping: Patience of 10 epochs

## Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve

### Cross-Dataset Evaluation
- Train on FaceForensics++, test on Celeb-DF
- Evaluate generalization across manipulation methods
- Test robustness under compression and noise

## Deployment

### Development Server
```bash
python app.py
```

### Production (Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```bash
docker build -t deepfake-detector .
docker run -p 5000:5000 deepfake-detector
```

## API Reference

### POST /upload
Upload video for detection
- **Input**: Multipart form data with video file
- **Output**: JSON with job_id

### GET /analyze/{job_id}
Analyze uploaded video
- **Input**: job_id from upload
- **Output**: JSON with prediction, confidence, visualizations

### GET /results/{job_id}
View results page
- **Output**: HTML page with detection results

### GET /health
Health check endpoint
- **Output**: System status and version

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Reduce sequence length
   - Use smaller backbone (efficientnet_b0)

2. **No Faces Detected**
   - Check video quality
   - Ensure face is visible and frontal
   - Adjust face detection confidence threshold

3. **Slow Inference**
   - Use GPU if available
   - Reduce number of frames
   - Use smaller model architecture

## Performance Optimization

- **Model Quantization**: Convert to INT8 for faster inference
- **ONNX Export**: Export model to ONNX format
- **Batch Processing**: Process multiple videos in parallel
- **Caching**: Cache preprocessed frames

## Future Enhancements

1. Audio analysis for multimodal detection
2. Real-time video streaming analysis
3. Mobile deployment (TensorFlow Lite)
4. Adversarial robustness improvements
5. Explainability improvements (LIME, SHAP)

## References

See project proposal for complete bibliography of research papers and datasets.
