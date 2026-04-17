"""
Multimodal Deepfake Detection System
Flask Application Entry Point

CPSC 589 - Graduate Project
California State University Fullerton
"""

import os
import uuid
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from config import config
import torch

def setup_logging(app):
    """Configure file and console logging with rotation"""
    # Create logs directory if it doesn't exist
    log_folder = app.config.get('LOG_FOLDER', 'logs')
    os.makedirs(log_folder, exist_ok=True)
    
    # Set logging level
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Main log file handler (all levels) with rotation (10MB per file, keep 5 backups)
    main_log_file = os.path.join(log_folder, 'deepfake_detection.log')
    file_handler = RotatingFileHandler(
        main_log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Error log file handler (ERROR and CRITICAL only)
    error_log_file = os.path.join(log_folder, 'errors.log')
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add all handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Log startup message
    app.logger.info("=" * 70)
    app.logger.info("Deepfake Detection System Started")
    app.logger.info(f"Log files location: {log_folder}")
    app.logger.info(f"Main log: {main_log_file}")
    app.logger.info(f"Error log: {error_log_file}")
    app.logger.info("=" * 70)

logger = logging.getLogger(__name__)

def create_app(config_name='development'):
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Setup logging with file handlers
    setup_logging(app)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() and app.config['USE_GPU'] else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Store detection results temporarily
    detection_results = {}
    detector_instance = None
    detector_ready = True  # BitMind API is ready (will verify on first use)
    
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    @app.route('/')
    def index():
        """Main page"""
        return render_template('index.html')
    
    @app.route('/about')
    def about():
        """About page"""
        return render_template('about.html')
    
    @app.route('/health')
    def health():
        """Health check endpoint"""
        # Check BitMind API availability
        try:
            from utils.bitmind_detector import BitMindDetector
            bitmind_api_key = app.config.get('BITMIND_API_KEY')
            bitmind_healthy = False
            if bitmind_api_key:
                detector = BitMindDetector(bitmind_api_key)
                bitmind_healthy = detector.is_healthy()
        except Exception as e:
            logger.warning(f"BitMind health check failed: {e}")
            bitmind_healthy = False
        
        return jsonify({
            'status': 'healthy' if bitmind_healthy else 'degraded',
            'version': app.config['VERSION'],
            'detector_ready': bitmind_healthy,  # For backward compatibility
            'detection_source': 'BitMind API',
            'bitmind_api_healthy': bitmind_healthy,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @app.route('/upload', methods=['POST'])
    def upload_video():
        """
        Handle video upload and initiate detection
        """
        try:
            # Check if file is present
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
            
            file = request.files['video']
            
            # Check if file is selected
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Validate file type
            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'Invalid file type. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
                }), 400
            
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            
            # Save file securely
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()
            saved_filename = f"{job_id}.{file_extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(filepath)
            
            logger.info(f"File uploaded: {saved_filename} (Job ID: {job_id})")
            
            # TODO: Process video asynchronously using Celery
            # For now, we'll return job_id and process synchronously in /analyze
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'filename': filename,
                'message': 'Video uploaded successfully. Processing...'
            }), 200
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    
    @app.route('/analyze/<job_id>', methods=['GET'])
    def analyze_video(job_id):
        """
        Analyze uploaded video and return detection results using BitMind API
        """
        try:
            # Import BitMind detector
            from utils.bitmind_detector import BitMindDetector
            
            # Initialize BitMind detector (lazy loading)
            nonlocal detector_instance
            if detector_instance is None:
                logger.info("Initializing BitMind API detector")
                bitmind_api_key = os.environ.get('BITMIND_API_KEY') or app.config.get('BITMIND_API_KEY')
                if not bitmind_api_key:
                    logger.error("BitMind API key not configured")
                    return jsonify({
                        'error': 'BitMind API key not configured',
                        'hint': 'Set BITMIND_API_KEY environment variable'
                    }), 503
                detector_instance = BitMindDetector(bitmind_api_key)
                logger.info("BitMind detector initialized successfully")
            
            # Find video file
            video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                          if f.startswith(job_id)]
            
            if not video_files:
                logger.error(f"Video file not found for job_id: {job_id}")
                return jsonify({'error': 'Video not found'}), 404
            
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
            
            logger.info(f"Analyzing video with BitMind API: {job_id}")
            
            # Call BitMind API for detection
            bitmind_result = detector_instance.detect_video(video_path, debug=False)
            
            # Handle API errors
            if bitmind_result.get('error'):
                logger.error(f"BitMind API error: {bitmind_result['error']}")
                return jsonify({
                    'error': bitmind_result['error'],
                    'job_id': job_id
                }), 400
            
            # Prepare results
            is_ai = bitmind_result.get('is_ai', False)
            confidence = bitmind_result.get('confidence', 0)
            
            result = {
                'job_id': job_id,
                'prediction': 'FAKE' if is_ai else 'REAL',
                'confidence': float(confidence),
                'detection_source': 'BitMind API',
                'is_ai_generated': is_ai,
                'similarity_score': bitmind_result.get('similarity', 0),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store result
            detection_results[job_id] = result
            
            logger.info(f"Analysis complete: {job_id} - Prediction: {result['prediction']} (confidence: {confidence:.2%})")
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/results/<job_id>')
    def get_results(job_id):
        """Get detection results"""
        if job_id in detection_results:
            return render_template('results.html', result=detection_results[job_id])
        else:
            return render_template('error.html', 
                                 error='Results not found. Please upload and analyze a video first.'), 404
    
    @app.route('/api/results/<job_id>')
    def api_get_results(job_id):
        """API endpoint to get results as JSON"""
        if job_id in detection_results:
            return jsonify(detection_results[job_id]), 200
        else:
            return jsonify({'error': 'Results not found'}), 404
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return render_template('error.html', error='Page not found'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        logger.error(f"Internal error: {str(error)}")
        return render_template('error.html', error='Internal server error'), 500
    
    return app


if __name__ == '__main__':
    # Create app with development config
    app = create_app('development')
    
    # Run Flask development server
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )
