import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Application Settings
    APP_NAME = 'Multimodal Deepfake Detection System'
    VERSION = '1.0.0'
    
    # File Upload Settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    
    # Model Settings
    MODEL_FOLDER = os.path.join(os.path.dirname(__file__), 'models', 'pretrained')
    SPATIAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'spatial_model.pth')
    TEMPORAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'temporal_model.pth')
    FUSION_MODEL_PATH = os.path.join(MODEL_FOLDER, 'fusion_model.pth')
    
    # Video Processing Settings
    FRAME_EXTRACTION_FPS = 10  # Extract 10 frames per second
    MAX_FRAMES = 300  # Maximum frames to process
    FRAME_SIZE = (224, 224)  # Input size for CNN
    SEQUENCE_LENGTH = 30  # Number of frames for temporal analysis
    
    # Face Detection Settings
    FACE_DETECTION_CONFIDENCE = 0.5
    FACE_MARGIN = 0.2  # 20% margin around detected face
    
    # Model Hyperparameters
    BATCH_SIZE = 16
    NUM_CLASSES = 2  # Real vs Fake
    CONFIDENCE_THRESHOLD = 0.5
    
    # Training Settings
    LEARNING_RATE = 0.0001
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    
    # Database Settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(__file__), 'deepfake_detection.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Celery Settings (for async tasks)
    CELERY_BROKER_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    
    # Security Settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Logging Settings
    LOG_FOLDER = os.path.join(os.path.dirname(__file__), 'logs')
    LOG_LEVEL = 'INFO'
    
    # Explainability Settings
    GRAD_CAM_LAYER = 'layer4'  # For ResNet-based models
    GENERATE_HEATMAPS = True
    GENERATE_TEMPORAL_PLOTS = True
    
    # Dataset Paths
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
    RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'raw')
    PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'processed')
    ANNOTATIONS_FOLDER = os.path.join(DATA_FOLDER, 'annotations')
    
    # Performance Settings
    USE_GPU = True
    NUM_WORKERS = 4  # DataLoader workers
    PIN_MEMORY = True
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
        os.makedirs(Config.LOG_FOLDER, exist_ok=True)
        os.makedirs(Config.DATA_FOLDER, exist_ok=True)
        os.makedirs(Config.RAW_DATA_FOLDER, exist_ok=True)
        os.makedirs(Config.PROCESSED_DATA_FOLDER, exist_ok=True)
        os.makedirs(Config.ANNOTATIONS_FOLDER, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Override with environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Use stronger security settings
    SESSION_COOKIE_SECURE = True
    

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
