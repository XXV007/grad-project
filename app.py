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
        return jsonify({
            'status': 'healthy',
            'version': app.config['VERSION'],
            'device': str(device),
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
        Analyze uploaded video and return detection results
        """
        try:
            # Import detection modules (lazy loading)
            from utils.preprocessing import VideoPreprocessor
            from models.fusion_model import SimpleMultimodalDetector
            from utils.explainability import ExplainabilityModule
            
            # Find video file
            video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                          if f.startswith(job_id)]
            
            if not video_files:
                return jsonify({'error': 'Video not found'}), 404
            
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
            
            logger.info(f"Analyzing video: {job_id}")
            
            # Step 1: Preprocess video
            preprocessor = VideoPreprocessor(app.config)
            frames, faces_detected = preprocessor.process_video(video_path)
            
            if frames is None or len(frames) == 0:
                return jsonify({
                    'error': 'No faces detected in video or video processing failed'
                }), 400
            
            # Step 2: Load detection model once and reuse it across requests.
            nonlocal detector_instance
            if detector_instance is None:
                logger.info("Initializing detector for the first analysis request")
                detector_instance = SimpleMultimodalDetector(app.config, device)
            detector = detector_instance
            
            # Step 3: Run detection
            prediction, confidence, spatial_features, temporal_features = detector.predict(frames)
            
            # Step 4: Generate explainability visualizations
            explainer = ExplainabilityModule(app.config, detector.model, device)
            heatmap_path, temporal_plot_path = explainer.generate_visualizations(
                frames, spatial_features, temporal_features, job_id
            )
            
            # Step 5: Prepare results
            result = {
                'job_id': job_id,
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'confidence': float(confidence),
                'frames_analyzed': len(frames),
                'faces_detected': faces_detected,
                'heatmap_url': url_for('static', filename=f'results/{job_id}_heatmap.png'),
                'temporal_plot_url': url_for('static', filename=f'results/{job_id}_temporal.png'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store result
            detection_results[job_id] = result
            
            logger.info(f"Analysis complete: {job_id} - Prediction: {result['prediction']}")
            
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
        debug=True
    )
