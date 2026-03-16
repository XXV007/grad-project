# Quick Start Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

## Installation Steps

### 1. Clone or Navigate to Project
```powershell
cd "c:\Users\vishn\OneDrive\Desktop\grad project"
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you encounter execution policy errors, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Some packages may require additional setup:
- **dlib**: Requires Visual Studio Build Tools
- **torch**: Install with CUDA support if you have an NVIDIA GPU:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### 4. Verify Installation
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Running the Application

### Development Mode
```powershell
python app.py
```

The application will start at `http://localhost:5000`

### Production Mode
```powershell
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

## Using the System

### 1. Upload Video
- Navigate to `http://localhost:5000`
- Click "Browse Files" or drag-and-drop a video
- Supported formats: MP4, AVI, MOV, MKV, WEBM
- Maximum size: 500MB

### 2. Analyze
- Click "Analyze Video"
- Wait for processing (may take 1-3 minutes depending on video length)
- System will:
  - Extract frames
  - Detect faces
  - Run spatial and temporal analysis
  - Generate explainability visualizations

### 3. View Results
- Results page shows:
  - Prediction (REAL or FAKE)
  - Confidence score
  - Number of frames analyzed
  - Spatial attention heatmap
  - Temporal anomaly plot

## Testing the Models

### Test Spatial Model
```powershell
python models\spatial_model.py
```

### Test Temporal Model
```powershell
python models\temporal_model.py
```

### Test Fusion Model
```powershell
python models\fusion_model.py
```

## Training Models (Optional)

### Train Spatial Model
```powershell
python training\train_spatial.py --backbone efficientnet_b0 --epochs 50
```

### Train Temporal Model
```powershell
python training\train_temporal.py --model_type lstm --epochs 50
```

## Troubleshooting

### Issue: Import errors
**Solution**: Ensure all dependencies are installed
```powershell
pip install -r requirements.txt
```

### Issue: MediaPipe not working
**Solution**: Fallback to Haar Cascade (automatic)
```powershell
pip uninstall mediapipe
```

### Issue: Slow processing
**Solution**: 
- Reduce video length
- Lower FPS in config.py (FRAME_EXTRACTION_FPS)
- Use GPU if available

### Issue: Port 5000 already in use
**Solution**: Use different port
```powershell
set PORT=8000
python app.py
```

### Issue: CUDA errors
**Solution**: Disable GPU in config.py
```python
USE_GPU = False
```

## Project Structure Overview

```
grad project/
├── app.py                  # Flask application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
│
├── models/                # ML models
│   ├── spatial_model.py   # CNN spatial extractor
│   ├── temporal_model.py  # LSTM/Transformer temporal analyzer
│   └── fusion_model.py    # Multimodal fusion
│
├── utils/                 # Utility modules
│   ├── preprocessing.py   # Video preprocessing
│   └── explainability.py  # Grad-CAM and visualizations
│
├── templates/             # HTML templates
│   ├── index.html        # Upload page
│   ├── results.html      # Results page
│   └── about.html        # About page
│
├── static/               # Static assets
│   ├── css/style.css     # Custom styles
│   └── js/main.js        # Frontend JavaScript
│
├── training/             # Training scripts
│   └── train_spatial.py  # Example training script
│
└── docs/                 # Documentation
    └── architecture.md   # System architecture guide
```

## Next Steps

1. **Collect Datasets**: Download FaceForensics++, DFDC, or Celeb-DF
2. **Train Models**: Use training scripts to train on your datasets
3. **Evaluate**: Test on validation sets and measure metrics
4. **Deploy**: Deploy to cloud platform (AWS, GCP, Azure)

## Getting Help

- Check documentation in `docs/` folder
- Review code comments in source files
- Test individual components before full integration

## Important Notes

⚠️ **Model Weights**: This project includes model architectures but NOT pretrained weights. You need to:
1. Download pretrained ImageNet weights (automatic with `pretrained=True`)
2. Train on deepfake datasets for actual detection capability
3. Or download pre-trained deepfake detection weights if available

⚠️ **Dataset**: You need to obtain deepfake datasets separately due to size and licensing.

⚠️ **GPU**: While the system works on CPU, GPU is highly recommended for:
- Training models
- Processing long videos
- Real-time analysis

## Success Indicators

✅ Application starts without errors
✅ Can upload and process test video
✅ Results page displays with visualizations
✅ No import errors when running model tests

## Contact

**Student**: Vishnu Priyan Bhaskar  
**Email**: vishnumax03@csu.fullerton.edu  
**Course**: CPSC 589 - California State University Fullerton
