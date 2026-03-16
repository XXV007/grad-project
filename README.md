# Multimodal and Robust Deepfake Detection System

**CPSC 589 - Graduate Project**  
**California State University Fullerton**  
**Student:** Vishnu Priyan Bhaskar (824838833)  
**Advisor:** Prof. Kenneth Kung

[![Status](https://img.shields.io/badge/Status-Implementation%20Complete-brightgreen)]()
[![Training](https://img.shields.io/badge/Training-In%20Progress-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.13%20tested-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)]()
[![Flask](https://img.shields.io/badge/Flask-3.1-black)]()

## 📋 Project Status

**Current Phase:** ✅ Implementation Complete | 🔄 Training & Evaluation In Progress

The implemented system **fully aligns** with the approved project proposal, integrating:
- ✅ Spatial feature extraction via CNNs (EfficientNet/XceptionNet/ResNet50)
- ✅ Temporal modeling through LSTM/Transformer architectures  
- ✅ Late-fusion strategy combining multimodal representations
- ✅ Explainability mechanisms (Grad-CAM + temporal visualizations)
- ✅ Production-ready web interface with comprehensive logging

**📊 See [PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md) for detailed implementation analysis**

---

## Project Overview

This project implements a **multimodal deepfake detection system** that combines spatial and temporal analysis techniques to identify synthetic or manipulated video content with high accuracy and robustness. The system integrates:

- **Spatial Analysis:** CNN-based frame-level feature extraction (EfficientNet-B0/B4, XceptionNet, ResNet50)
- **Temporal Analysis:** Sequence-based modeling with LSTM/Transformer for motion inconsistency detection
- **Late Fusion:** Multimodal combination via concatenation, addition, or attention mechanisms
- **Explainability:** Grad-CAM visualizations and temporal activation maps for transparent predictions
- **Web Interface:** Flask-based application with secure upload, inference, and result visualization

## ✨ Features

### Core Capabilities
- ✅ **Multimodal Architecture:** Spatial CNN (7.5M params) + Temporal LSTM/Transformer
- ✅ **Multiple Backbones:** EfficientNet-B0/B4, XceptionNet, ResNet50 (configurable)
- ✅ **Fusion Strategies:** Concatenation, addition, attention-based fusion
- ✅ **Explainable AI:** Grad-CAM heatmaps + temporal activation visualizations
- ✅ **Web Interface:** Professional Bootstrap 5 UI with drag-and-drop upload
- ✅ **Real-time Inference:** Confidence scoring and anomaly detection
- ✅ **Comprehensive Logging:** Rotating file logs with interactive viewer
- ✅ **Production Ready:** Dev/Prod configs, error handling, security features

### Dataset Support
- ✅ FaceForensics++ (FF++)
- ✅ Deepfake Detection Challenge (DFDC)
- ✅ Celeb-DF v2
- ✅ Custom dataset integration via PyTorch DataLoader

### Preprocessing Pipeline
- ✅ Face detection (MediaPipe + Haar Cascade fallback)
- ✅ Frame sampling (uniform/random, 30 frames default)
- ✅ Face alignment and cropping
- ✅ Normalization (ImageNet statistics)

## System Architecture

```
Input Video → Frame Extraction → Face Detection/Alignment
                                         ↓
                    ┌────────────────────┴────────────────────┐
                    ↓                                         ↓
            Spatial Analysis (CNN)                  Temporal Analysis (LSTM/3D-CNN)
                    ↓                                         ↓
                    └────────────────────┬────────────────────┘
                                         ↓
                              Feature Fusion (Late Fusion)
                                         ↓
                              Classification (Real/Fake)
                                         ↓
                    Explainability Module (Grad-CAM + Heatmaps)
```

## Installation

### Prerequisites
- Python 3.13 recommended for the current Windows demo setup
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd "grad project"
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

The checked-in requirements file is aligned to the currently working Flask demo path on Windows and Python 3.13. It focuses on the packages needed to run the app, upload a video, and generate the current explainability outputs.

4. Download pretrained models (if available):
```bash
python scripts/download_models.py
```

## Project Structure

```
grad project/
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings
├── README.md                   # Project documentation
│
├── models/                     # Machine learning models
│   ├── __init__.py
│   ├── spatial_model.py       # CNN spatial feature extractor
│   ├── temporal_model.py      # LSTM/3D-CNN temporal analyzer
│   ├── fusion_model.py        # Multimodal fusion architecture
│   └── pretrained/            # Pretrained model weights
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── preprocessing.py       # Video preprocessing & face detection
│   ├── data_loader.py         # Dataset loading utilities
│   ├── explainability.py      # Grad-CAM & visualization
│   └── metrics.py             # Evaluation metrics
│
├── training/                   # Training scripts
│   ├── train_spatial.py       # Train spatial model
│   ├── train_temporal.py      # Train temporal model
│   ├── train_fusion.py        # Train fusion model
│   └── evaluate.py            # Model evaluation
│
├── static/                     # Frontend static files
│   ├── css/
│   │   └── style.css          # Custom styles
│   ├── js/
│   │   └── main.js            # Frontend JavaScript
│   └── uploads/               # Temporary video uploads
│
├── templates/                  # HTML templates
│   ├── index.html             # Main upload page
│   ├── results.html           # Detection results page
│   └── about.html             # About page
│
├── data/                       # Dataset directory
│   ├── raw/                   # Raw video files
│   ├── processed/             # Preprocessed frames
│   └── annotations/           # Dataset labels
│
├── notebooks/                  # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation_analysis.ipynb
│
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── api_documentation.md
│   └── deployment_guide.md
│
└── tests/                      # Unit tests
    ├── test_models.py
    ├── test_preprocessing.py
    └── test_app.py
```

## Usage

### Running the Web Application

```bash
python app.py
```

Navigate to `http://localhost:5000` in your browser.

### Training Models

```bash
# Train spatial model
python training/train_spatial.py --dataset faceforensics --epochs 50

# Train temporal model
python training/train_temporal.py --dataset faceforensics --epochs 50

# Train fusion model
python training/train_fusion.py --dataset faceforensics --epochs 30
```

### Evaluating Models

```bash
python training/evaluate.py --model fusion --dataset celeb-df
```

## API Endpoints

- `POST /upload` - Upload video for detection
- `GET /results/<job_id>` - Get detection results
- `GET /visualize/<job_id>` - Get explainability visualizations
- `GET /health` - System health check

## Datasets Supported

- **FaceForensics++**: High-quality face manipulation dataset
- **DFDC (Deepfake Detection Challenge)**: Large-scale detection dataset
- **Celeb-DF**: Celebrity deepfake dataset
- **ForgeryNet**: Multi-method forgery dataset

## Performance Metrics

The system is evaluated using:
- Accuracy
- Precision & Recall
- F1-Score
- AUC-ROC
- Cross-dataset generalization
- Robustness under compression/noise

## Technologies Used

- **Backend:** Flask, Python
- **Deep Learning:** PyTorch, TensorFlow
- **Computer Vision:** OpenCV, MediaPipe, dlib
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap
- **Database:** SQLite/PostgreSQL
- **Deployment:** Docker, AWS/GCP

## Ethical Considerations

This project handles biometric facial data responsibly:
- ✅ Secure data transmission (HTTPS/TLS)
- ✅ Minimal data retention policies
- ✅ Anonymization where applicable
- ✅ Compliance with privacy regulations

## Project Timeline

- **Weeks 1-2:** Project planning and literature review
- **Weeks 3-4:** Data collection and preprocessing
- **Weeks 5-6:** Algorithm implementation
- **Weeks 7-8:** Model integration and UI development
- **Weeks 9-10:** Evaluation and fine-tuning
- **Weeks 11-12:** Documentation and final presentation

## References

See [docs/references.md](docs/references.md) for complete bibliography.

## License

This project is for academic purposes as part of CPSC 589 coursework.

## Contact

**Vishnu Priyan Bhaskar**  
Email: vishnumax03@csu.fullerton.edu  
Student ID: 824838833

## Acknowledgments

- Prof. Kenneth Kung - Faculty Advisor
- CSUF Department of Computer Science
- Dataset providers (FaceForensics++, DFDC, Celeb-DF)
