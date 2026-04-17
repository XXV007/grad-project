"""
BitMind API Integration Module
Handles communication with BitMind deepfake detection API

BitMind API Documentation: https://api.bitmind.ai
"""

import requests
import os
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class BitMindDetector:
    """Wrapper for BitMind API deepfake detection"""
    
    def __init__(self, api_key: str):
        """
        Initialize BitMind detector
        
        Args:
            api_key: BitMind API key (format: key:secret)
        """
        self.api_key = api_key
        self.base_url = "https://api.bitmind.ai"
        self.video_endpoint = f"{self.base_url}/detect-video"
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }
        logger.info("BitMind detector initialized")
    
    def detect_video(self, video_path: str, debug: bool = False) -> Dict:
        """
        Detect deepfake in video using BitMind API
        
        Args:
            video_path: Path to video file
            debug: Include debug data in response
        
        Returns:
            Dict with keys:
            - is_ai: bool (True if AI-generated/deepfake)
            - confidence: float (0-1)
            - similarity: float (0-1)
            - object_key: str (BitMind storage key)
            - error: str (if error occurred)
        """
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return {
                    'error': f'Video file not found: {video_path}',
                    'is_ai': None,
                    'confidence': 0
                }
            
            # Get file size
            file_size = os.path.getsize(video_path)
            logger.info(f"Sending video to BitMind API: {video_path} ({file_size / 1024 / 1024:.2f} MB)")
            
            # Prepare multipart form data
            with open(video_path, 'rb') as video_file:
                files = {
                    'video': ('video.mp4', video_file, 'video/mp4')
                }
                data = {
                    'debug': 'true' if debug else 'false'
                }
                
                # Make request to BitMind API
                logger.info("Sending request to BitMind API...")
                response = requests.post(
                    self.video_endpoint,
                    headers=self.headers,
                    files=files,
                    data=data,
                    timeout=300  # 5 minute timeout for video processing
                )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                logger.info(f"BitMind API response: {result}")
                
                return {
                    'is_ai': result.get('isAI', False),
                    'confidence': result.get('confidence', 0),
                    'similarity': result.get('similarity', 0),
                    'object_key': result.get('objectKey', ''),
                    'thumbnail_key': result.get('thumbnailObjectKey', ''),
                    'error': None
                }
            
            elif response.status_code == 401:
                error_msg = "Invalid BitMind API key"
                logger.error(f"BitMind API Error: {error_msg}")
                return {
                    'error': error_msg,
                    'is_ai': None,
                    'confidence': 0
                }
            
            elif response.status_code == 429:
                error_msg = "BitMind API rate limit exceeded. Please try again later."
                logger.error(f"BitMind API Error: {error_msg}")
                return {
                    'error': error_msg,
                    'is_ai': None,
                    'confidence': 0
                }
            
            else:
                error_msg = f"BitMind API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'error': error_msg,
                    'is_ai': None,
                    'confidence': 0
                }
        
        except requests.exceptions.Timeout:
            error_msg = "BitMind API request timeout. Video processing took too long."
            logger.error(error_msg)
            return {
                'error': error_msg,
                'is_ai': None,
                'confidence': 0
            }
        
        except requests.exceptions.ConnectionError:
            error_msg = "Failed to connect to BitMind API. Check your internet connection."
            logger.error(error_msg)
            return {
                'error': error_msg,
                'is_ai': None,
                'confidence': 0
            }
        
        except Exception as e:
            error_msg = f"BitMind API error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'error': error_msg,
                'is_ai': None,
                'confidence': 0
            }
    
    def is_healthy(self) -> bool:
        """Check if BitMind API is accessible"""
        try:
            # Simple health check by testing with a dummy request
            logger.info("Checking BitMind API health...")
            response = requests.head(
                self.base_url,
                headers=self.headers,
                timeout=5
            )
            is_healthy = response.status_code < 500
            logger.info(f"BitMind API health: {'OK' if is_healthy else 'DEGRADED'}")
            return is_healthy
        except Exception as e:
            logger.error(f"BitMind health check failed: {e}")
            return False
