# 📝 Logging System Documentation

## Overview

Your Deepfake Detection System now has **comprehensive error logging** with both file and console output. All operations, errors, and system events are automatically tracked.

---

## ✅ What's Included

### 1. **Dual Log Files**
- **Main Log** (`logs/deepfake_detection.log`): All events (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Error Log** (`logs/errors.log`): Only errors and critical issues for quick debugging

### 2. **Automatic Log Rotation**
- Files automatically rotate when they reach **10MB**
- Keeps **5 backup files** (e.g., `.log.1`, `.log.2`, etc.)
- Oldest backups are deleted automatically
- Never worry about disk space!

### 3. **Interactive Log Viewer** (`view_logs.py`)
- Easy-to-use menu system
- Search functionality
- Statistical analysis
- Filter by time period

### 4. **Detailed Logging Format**
Each log entry includes:
```
2026-01-28 14:30:45 - module_name - ERROR - [file.py:123] - Error message here
     ↓                   ↓            ↓           ↓              ↓
  Timestamp         Module Name    Level    File:Line      Detailed Message
```

---

## 📊 What Gets Logged

### ✅ Successfully Logged Events

| Event Type | Log Level | Example |
|------------|-----------|---------|
| Application startup | INFO | "Deepfake Detection System Started" |
| Video uploads | INFO | "File uploaded: video.mp4 (Job ID: abc-123)" |
| Analysis start | INFO | "Analyzing video: abc-123" |
| Analysis completion | INFO | "Analysis complete: abc-123 - Prediction: FAKE" |
| Device selection | INFO | "Using device: cuda:0" |
| Face detection failures | WARNING | "No faces detected in video" |
| Upload errors | ERROR | "Upload error: Invalid file type" |
| Analysis errors | ERROR | "Analysis error: Model inference failed" |
| Internal server errors | ERROR | "Internal error: Database connection lost" |
| All stack traces | ERROR | Full traceback for debugging |

---

## 🚀 How to Use

### Method 1: Interactive Log Viewer (Recommended)

```powershell
# Navigate to project folder
cd "c:\Users\vishn\OneDrive\Desktop\grad project"

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run interactive viewer
python view_logs.py
```

**Menu Options:**
```
1. Show last 50 lines of main log
2. Show last 100 lines of main log
3. Show all errors (last 24 hours)
4. Show all errors (last 7 days)
5. Search logs for specific term
6. Analyze log statistics
7. Show error log only
0. Exit
```

### Method 2: Command Line

```powershell
# Show last 50 lines
python view_logs.py tail 50

# Show errors from last 24 hours
python view_logs.py errors 24

# Search for specific term
python view_logs.py search "FAKE"

# Show statistics
python view_logs.py stats
```

### Method 3: PowerShell Direct Access

```powershell
# View recent main log
Get-Content logs\deepfake_detection.log -Tail 50

# View recent errors
Get-Content logs\errors.log -Tail 20

# Follow logs in real-time
Get-Content logs\deepfake_detection.log -Wait -Tail 20

# Search for specific job ID
Select-String -Path logs\deepfake_detection.log -Pattern "abc-123"

# Count errors today
Select-String -Path logs\errors.log -Pattern (Get-Date -Format "yyyy-MM-dd")
```

---

## 🔍 Common Use Cases

### 1. **Check if System is Working**
```powershell
python view_logs.py tail 20
```
Look for recent INFO messages indicating normal operation.

### 2. **Debug Failed Upload**
```powershell
python view_logs.py search "Upload error"
```
Shows all upload failures with reasons.

### 3. **Find Errors in Last Hour**
```powershell
python view_logs.py errors 1
```
Quick check for recent problems.

### 4. **Track Specific Video Analysis**
```powershell
python view_logs.py search "job-id-here"
```
Shows complete timeline for one video.

### 5. **System Health Check**
```powershell
python view_logs.py stats
```
Shows:
- Total operations by type
- Error rates
- Success rates
- Module activity

---

## 📈 Log Statistics Example

```
📊 Log Statistics
====================================================================

📈 By Log Level:
  DEBUG      :   143 ███████████████
  INFO       :   856 ████████████████████████████████████████████████
  WARNING    :    23 ██
  ERROR      :    12 █
  CRITICAL   :     0 

📦 By Module (Top 10):
  app                  :   445 ████████████████████████
  utils.preprocessing  :   234 ████████████
  models.fusion_model  :   156 ████████
  utils.explainability :   134 ███████

🎯 Operations:
  Videos Uploaded  : 45
  Videos Analyzed  : 42
  Success Rate     : 93.3%
```

---

## 🛡️ Security & Privacy

### ✅ What's Safe
- Logs are **excluded from Git** (in `.gitignore`)
- No sensitive data logged (passwords, API keys)
- Job IDs are random UUIDs (not traceable)

### ⚠️ Be Careful
- Error logs may contain file paths
- Stack traces show code structure
- Don't share logs publicly without review

---

## 🔧 Configuration

### Change Log Level (in `config.py`)

```python
# Development (verbose)
LOG_LEVEL = 'DEBUG'  # Shows everything

# Production (quiet)
LOG_LEVEL = 'WARNING'  # Only warnings and errors

# Error-only (minimal)
LOG_LEVEL = 'ERROR'  # Only errors and critical
```

### Change Log Location (in `config.py`)

```python
LOG_FOLDER = os.path.join(os.path.dirname(__file__), 'logs')  # Default
# or
LOG_FOLDER = 'C:/custom/path/to/logs'  # Custom path
```

### Adjust Rotation Settings (in `app.py`)

```python
# In setup_logging() function:

# Increase file size before rotation
maxBytes=50*1024*1024,  # 50MB instead of 10MB

# Keep more backups
backupCount=10  # 10 files instead of 5
```

---

## 🐛 Troubleshooting

### Problem: No log files created

**Solution:**
1. Check if `logs/` folder exists
2. Verify write permissions
3. Ensure application has started (run `python app.py`)

### Problem: Log files too large

**Solution:**
1. Check rotation settings (should auto-rotate at 10MB)
2. Reduce LOG_LEVEL to 'WARNING' or 'ERROR'
3. Manually delete old `.log.X` backup files

### Problem: Can't find specific error

**Solution:**
1. Use search: `python view_logs.py search "error term"`
2. Check `errors.log` instead of main log
3. Verify date range (errors might be in older backup files)

### Problem: Logs not showing in view_logs.py

**Solution:**
1. Make sure application has run at least once
2. Check that log files exist in `logs/` folder
3. Verify you're in the correct directory

---

## 📚 Quick Command Reference

| Task | Command |
|------|---------|
| Start interactive viewer | `python view_logs.py` |
| Show last 50 lines | `python view_logs.py tail 50` |
| Show errors (24h) | `python view_logs.py errors 24` |
| Search logs | `python view_logs.py search "term"` |
| Show statistics | `python view_logs.py stats` |
| Follow in real-time | `Get-Content logs\deepfake_detection.log -Wait -Tail 20` |
| View errors only | `Get-Content logs\errors.log` |
| Search by job ID | `Select-String -Path logs\*.log -Pattern "job-id"` |

---

## 🎯 Best Practices

1. **Check logs regularly** - Monitor system health
2. **Review errors weekly** - Identify patterns
3. **Clean old logs monthly** - Save disk space (delete `.log.X` files)
4. **Use stats feature** - Track success rates
5. **Search before debugging** - Logs often have the answer
6. **Follow in production** - `Get-Content -Wait` for live monitoring

---

## 📖 Related Documentation

- **Quick Reference**: `QUICK_REFERENCE.md`
- **Setup Guide**: `QUICKSTART.md`
- **Full Verification**: `FINAL_VERIFICATION.md`
- **Log Directory Info**: `logs/README.md`

---

## ✅ Summary

✅ **Automatic logging** - No manual setup needed  
✅ **Dual files** - Main log + error-only log  
✅ **Auto-rotation** - Never runs out of space  
✅ **Easy viewing** - Interactive + command-line tools  
✅ **Searchable** - Find anything quickly  
✅ **Statistical analysis** - Track system health  
✅ **Production-ready** - Safe for deployment  

**Your system is now fully instrumented for debugging and monitoring!** 🎉
