"""
System Verification Test Script
Tests all major components of the deepfake detection system

Run this to verify installation and functionality
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test if all required packages can be imported"""
    print("\n" + "="*60)
    print("TESTING PACKAGE IMPORTS")
    print("="*60)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'flask': 'Flask',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn'
    }
    
    results = {}
    for package, name in packages.items():
        try:
            __import__(package)
            results[name] = '✓ Installed'
            print(f"  {name:<20} ✓")
        except ImportError:
            results[name] = '✗ Missing'
            print(f"  {name:<20} ✗ MISSING")
    
    return results


def test_project_structure():
    """Test if all required directories and files exist"""
    print("\n" + "="*60)
    print("TESTING PROJECT STRUCTURE")
    print("="*60)
    
    required_files = [
        'app.py',
        'config.py',
        'requirements.txt',
        'models/__init__.py',
        'models/spatial_model.py',
        'models/temporal_model.py',
        'models/fusion_model.py',
        'utils/__init__.py',
        'utils/preprocessing.py',
        'utils/explainability.py',
        'utils/metrics.py',
        'utils/data_loader.py',
        'templates/index.html',
        'templates/results.html',
        'static/css/style.css',
        'static/js/main.js'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  {file_path:<40} ✓")
        else:
            print(f"  {file_path:<40} ✗ MISSING")
            missing.append(file_path)
    
    return missing


def test_config():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    
    try:
        from config import config, DevelopmentConfig
        
        dev_config = config['development']
        
        print(f"  Configuration class: ✓")
        print(f"  FRAME_SIZE: {dev_config.FRAME_SIZE}")
        print(f"  SEQUENCE_LENGTH: {dev_config.SEQUENCE_LENGTH}")
        print(f"  BATCH_SIZE: {dev_config.BATCH_SIZE}")
        print(f"  USE_GPU: {dev_config.USE_GPU}")
        
        return True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False


def test_models():
    """Test model instantiation"""
    print("\n" + "="*60)
    print("TESTING MODEL INSTANTIATION")
    print("="*60)
    
    try:
        import torch
        from models.spatial_model import SpatialFeatureExtractor
        from models.temporal_model import TemporalAnalyzer
        from models.fusion_model import MultimodalDetector
        
        # Test spatial model
        print("  Testing Spatial Model...")
        spatial_model = SpatialFeatureExtractor(
            backbone='efficientnet_b0',
            pretrained=False
        )
        print(f"    ✓ Spatial model created")
        print(f"    Parameters: {sum(p.numel() for p in spatial_model.parameters()):,}")
        
        # Test temporal model
        print("  Testing Temporal Model...")
        temporal_model = TemporalAnalyzer(
            input_dim=1280,
            hidden_dim=512,
            model_type='lstm'
        )
        print(f"    ✓ Temporal model created")
        print(f"    Parameters: {sum(p.numel() for p in temporal_model.parameters()):,}")
        
        # Test fusion model
        print("  Testing Fusion Model...")
        fusion_model = MultimodalDetector(
            spatial_backbone='efficientnet_b0',
            temporal_type='lstm'
        )
        print(f"    ✓ Fusion model created")
        print(f"    Parameters: {sum(p.numel() for p in fusion_model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device():
    """Test CUDA availability"""
    print("\n" + "="*60)
    print("TESTING DEVICE CONFIGURATION")
    print("="*60)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
            print(f"  Device Count: {torch.cuda.device_count()}")
        else:
            print(f"  Using CPU (GPU recommended for training)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Device error: {e}")
        return False


def test_flask_app():
    """Test Flask application creation"""
    print("\n" + "="*60)
    print("TESTING FLASK APPLICATION")
    print("="*60)
    
    try:
        from app import create_app
        
        app = create_app('development')
        print(f"  ✓ Flask app created")
        print(f"  Debug mode: {app.config['DEBUG']}")
        print(f"  Max upload size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f} MB")
        
        # Test routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        print(f"  Routes registered: {len(routes)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Flask app error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test model forward pass with dummy data"""
    print("\n" + "="*60)
    print("TESTING MODEL FORWARD PASS")
    print("="*60)
    
    try:
        import torch
        from models.fusion_model import MultimodalDetector
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {device}")
        
        model = MultimodalDetector(
            spatial_backbone='efficientnet_b0',
            temporal_type='lstm'
        ).to(device)
        
        # Create dummy input
        batch_size = 1
        num_frames = 30
        C, H, W = 3, 224, 224
        
        dummy_input = torch.randn(batch_size, num_frames, C, H, W).to(device)
        
        print(f"  Input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            logits, spatial_feat, temporal_feat = model(dummy_input)
        
        print(f"  ✓ Forward pass successful")
        print(f"    Output logits: {logits.shape}")
        print(f"    Spatial features: {spatial_feat.shape}")
        print(f"    Temporal features: {temporal_feat.shape}")
        
        # Test prediction
        prediction, confidence, _, _ = model.predict(dummy_input)
        print(f"  ✓ Prediction successful")
        print(f"    Prediction: {prediction} (0=Real, 1=Fake)")
        print(f"    Confidence: {confidence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all verification tests"""
    print("\n" + "="*70)
    print(" DEEPFAKE DETECTION SYSTEM - VERIFICATION TESTS")
    print("="*70)
    
    results = {}
    
    # Run tests
    results['Imports'] = test_imports()
    results['Structure'] = len(test_project_structure()) == 0
    results['Config'] = test_config()
    results['Models'] = test_models()
    results['Device'] = test_device()
    results['Flask App'] = test_flask_app()
    results['Forward Pass'] = test_forward_pass()
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        if isinstance(result, dict):
            # For imports test
            status = all('✓' in v for v in result.values())
        else:
            status = result
        
        symbol = "✓" if status else "✗"
        print(f"  {test_name:<20} {symbol}")
    
    all_passed = all(
        (all('✓' in v for v in result.values()) if isinstance(result, dict) else result)
        for result in results.values()
    )
    
    print("="*70)
    if all_passed:
        print("\n✓ ALL TESTS PASSED! System is ready to use.")
        print("\nNext steps:")
        print("  1. Install missing packages if any: pip install -r requirements.txt")
        print("  2. Run the application: python app.py")
        print("  3. Open browser: http://localhost:5000")
    else:
        print("\n✗ SOME TESTS FAILED. Please review errors above.")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check Python version: python --version (requires 3.8+)")
        print("  3. Check CUDA if using GPU: nvidia-smi")
    
    print("\n")
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
