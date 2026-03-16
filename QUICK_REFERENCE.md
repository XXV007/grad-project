# 🚀 Quick Reference Card

## Multimodal Deepfake Detection System
**CPSC 589 | California State University Fullerton**

---

## ⚡ Quick Start (3 Steps)

```powershell
# 1. Setup Environment
cd "c:\Users\vishn\OneDrive\Desktop\grad project"
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install Dependencies  
pip install -r requirements.txt

# 3. Run Application
python app.py
```

Open: `http://localhost:5000`

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application |
| `config.py` | All settings |
| `test_system.py` | Verify installation |
| `QUICKSTART.md` | Detailed setup guide |
| `FINAL_VERIFICATION.md` | Complete verification |

---

## 🔧 Common Commands

```powershell
# Test system
python test_system.py

# Run app (development)
python app.py

# Run app (production)
waitress-serve --host=0.0.0.0 --port=5000 app:app

# View logs (interactive)
python view_logs.py

# View logs (command line)
python view_logs.py tail 50        # Last 50 lines
python view_logs.py errors 24      # Errors in last 24 hours
python view_logs.py search "ERROR" # Search for term
python view_logs.py stats          # Show statistics

# Test models
python models/spatial_model.py
python models/temporal_model.py
python models/fusion_model.py
```

---

## 🎯 System Architecture

```
Video → Preprocess → Spatial CNN ──┐
                                    ├→ Fusion → Prediction
Video → Preprocess → Temporal LSTM ─┘
```

---

## 📊 Model Components

| Component | Options |
|-----------|---------|
| **Spatial** | EfficientNet, Xception, ResNet50 |
| **Temporal** | LSTM, Transformer |
| **Fusion** | Concat, Add, Attention |

---

## ⚙️ Key Configuration

```python
# config.py
FRAME_SIZE = (224, 224)
SEQUENCE_LENGTH = 30
BATCH_SIZE = 16
MAX_CONTENT_LENGTH = 500 MB
USE_GPU = True
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| Port in use | `$env:PORT="8000"; python app.py` |
| CUDA errors | Set `USE_GPU = False` in config.py |
| Memory error | Reduce `BATCH_SIZE` or `SEQUENCE_LENGTH` |

---

## 📈 Expected Performance

- **Accuracy**: 85-95%
- **Inference**: 2-5 seconds (GPU)
- **Video Support**: MP4, AVI, MOV, MKV, WEBM
- **Max Size**: 500MB

---

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed
- [ ] `test_system.py` passes
- [ ] App runs on localhost:5000
- [ ] Can upload video
- [ ] Results page displays

---

## 📚 Documentation

1. **QUICKSTART.md** - Setup instructions
2. **PROJECT_GUIDE.md** - Complete guide
3. **FINAL_VERIFICATION.md** - Verification report
4. **docs/architecture.md** - Technical details

---

## 🎓 Project Info

- **Course**: CPSC 589
- **Student**: Vishnu Priyan Bhaskar (824838833)
- **Advisor**: Prof. Kenneth Kung
- **Institution**: CSUF Computer Science

---

## 🏆 Features

✅ Multimodal detection  
✅ Spatial + Temporal analysis  
✅ Explainable AI (Grad-CAM)  
✅ Web interface  
✅ Real-time processing  
✅ Beautiful visualizations  

---

## 🚨 Important Notes

⚠️ **Model Weights**: Train models or download pretrained weights  
⚠️ **Datasets**: Download FaceForensics++, DFDC, or Celeb-DF  
⚠️ **GPU**: Highly recommended for training and inference  

---

## 📞 Support

Check documentation:
- `QUICKSTART.md`
- `FINAL_VERIFICATION.md`
- Run: `python test_system.py`

---

## 📊 Logging & Monitoring

### Log Files Location
```
logs/
├── deepfake_detection.log    # Main application log (all levels)
└── errors.log                 # Error-only log (quick debugging)
```

### View Logs
```powershell
# Interactive log viewer
python view_logs.py

# View recent logs
Get-Content logs\deepfake_detection.log -Tail 50

# Follow logs in real-time
Get-Content logs\deepfake_detection.log -Wait -Tail 20

# View errors only
Get-Content logs\errors.log

# Search for specific job
Select-String -Path logs\deepfake_detection.log -Pattern "job-id-here"
```

### Log Levels
- **DEBUG**: Detailed diagnostic info
- **INFO**: Normal operations
- **WARNING**: Potential issues
- **ERROR**: Failures
- **CRITICAL**: Severe problems

### What Gets Logged
- ✅ Video uploads with job IDs
- ✅ Analysis start/completion
- ✅ Prediction results
- ✅ All errors with stack traces
- ✅ Device usage (GPU/CPU)
- ✅ Face detection results
```
