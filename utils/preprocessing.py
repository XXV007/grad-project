"""
Video Preprocessing Module
Handles video loading, frame extraction, face detection, and alignment

CPSC 589 - Multimodal Deepfake Detection
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """
    Video preprocessing pipeline for deepfake detection
    """
    
    def __init__(self, config):
        """
        Initialize video preprocessor
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.frame_size = config.get('FRAME_SIZE', (224, 224))
        self.fps = config.get('FRAME_EXTRACTION_FPS', 10)
        self.max_frames = config.get('MAX_FRAMES', 300)
        self.sequence_length = config.get('SEQUENCE_LENGTH', 30)
        
        # Initialize face detector
        self.face_detector = self._init_face_detector()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _init_face_detector(self):
        """Initialize face detection model"""
        try:
            # Try to use MediaPipe (more accurate)
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            return mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=0.5
            )
        except Exception as e:
            logger.warning(f"MediaPipe not available: {e}. Falling back to Haar Cascade")
            # Fallback to OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
    
    def process_video(self, video_path):
        """
        Process video file: extract frames and detect faces
        
        Args:
            video_path: Path to video file
        
        Returns:
            frames: Tensor of preprocessed frames (num_frames, C, H, W)
            faces_detected: Number of faces detected
        """
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Extract frames
            raw_frames = self._extract_frames(video_path)
            
            if raw_frames is None or len(raw_frames) == 0:
                logger.error("No frames extracted from video")
                return None, 0
            
            logger.info(f"Extracted {len(raw_frames)} frames")
            
            # Detect and crop faces
            face_frames, faces_detected = self._detect_and_crop_faces(raw_frames)
            
            if face_frames is None or len(face_frames) == 0:
                logger.warning("No faces detected in video")
                return None, 0
            
            logger.info(f"Detected faces in {len(face_frames)} frames")
            
            # Preprocess frames
            processed_frames = self._preprocess_frames(face_frames)
            
            # Sample frames to sequence length
            sampled_frames = self._sample_frames(processed_frames, self.sequence_length)
            
            logger.info(f"Final processed frames: {sampled_frames.shape}")
            
            return sampled_frames, faces_detected
            
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            return None, 0
    
    def _extract_frames(self, video_path):
        """
        Extract frames from video at specified FPS
        
        Args:
            video_path: Path to video file
        
        Returns:
            frames: List of numpy arrays (H, W, C)
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video: {original_fps:.2f} FPS, {total_frames} total frames")
            
            # Calculate frame skip
            frame_skip = max(1, int(original_fps / self.fps))
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified intervals
                if frame_count % frame_skip == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                    
                    # Stop if max frames reached
                    if extracted_count >= self.max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames if len(frames) > 0 else None
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return None
    
    def _detect_and_crop_faces(self, frames):
        """
        Detect faces and crop regions
        
        Args:
            frames: List of frames (H, W, C)
        
        Returns:
            face_frames: List of cropped face regions
            faces_detected: Number of faces detected
        """
        face_frames = []
        faces_detected = 0
        
        try:
            import mediapipe as mp
            use_mediapipe = True
        except:
            use_mediapipe = False
        
        for frame in frames:
            face_crop = None
            
            if use_mediapipe:
                # MediaPipe face detection
                results = self.face_detector.process(frame)
                
                if results.detections:
                    detection = results.detections[0]  # Use first detected face
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    # Convert relative coordinates to absolute
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    
                    # Add margin
                    margin = int(0.2 * max(width, height))
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    width = min(w - x, width + 2 * margin)
                    height = min(h - y, height + 2 * margin)
                    
                    face_crop = frame[y:y+height, x:x+width]
                    faces_detected += 1
            else:
                # Haar Cascade face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    # Use largest face
                    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                    
                    # Add margin
                    margin = int(0.2 * max(w, h))
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(frame.shape[1] - x, w + 2 * margin)
                    h = min(frame.shape[0] - y, h + 2 * margin)
                    
                    face_crop = frame[y:y+h, x:x+w]
                    faces_detected += 1
            
            # If face detected, add to list; otherwise use full frame
            if face_crop is not None and face_crop.size > 0:
                face_frames.append(face_crop)
            else:
                face_frames.append(frame)
        
        return face_frames if len(face_frames) > 0 else None, faces_detected
    
    def _preprocess_frames(self, frames):
        """
        Preprocess frames with normalization
        
        Args:
            frames: List of numpy arrays
        
        Returns:
            tensor: Preprocessed frames tensor (N, C, H, W)
        """
        processed = []
        
        for frame in frames:
            try:
                # Apply transforms
                tensor = self.transform(frame)
                processed.append(tensor)
            except Exception as e:
                logger.warning(f"Error preprocessing frame: {e}")
                continue
        
        if len(processed) == 0:
            return None
        
        # Stack into single tensor
        return torch.stack(processed)
    
    def _sample_frames(self, frames, target_length):
        """
        Sample frames to target sequence length
        
        Args:
            frames: Tensor of frames (N, C, H, W)
            target_length: Desired sequence length
        
        Returns:
            sampled_frames: Tensor (target_length, C, H, W)
        """
        num_frames = frames.shape[0]
        
        if num_frames == target_length:
            return frames
        elif num_frames < target_length:
            # Repeat frames if too few
            repeat_factor = (target_length // num_frames) + 1
            frames = frames.repeat(repeat_factor, 1, 1, 1)
            return frames[:target_length]
        else:
            # Sample uniformly if too many
            indices = np.linspace(0, num_frames - 1, target_length, dtype=int)
            return frames[indices]


if __name__ == '__main__':
    # Test video preprocessor
    print("Video Preprocessor Test")
    print("This would normally process a video file")
    print("Example usage:")
    print("  preprocessor = VideoPreprocessor(config)")
    print("  frames, faces = preprocessor.process_video('video.mp4')")
