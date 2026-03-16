"""
Models package for Multimodal Deepfake Detection System
"""

from .spatial_model import SpatialFeatureExtractor
from .temporal_model import TemporalAnalyzer
from .fusion_model import MultimodalDetector

__all__ = ['SpatialFeatureExtractor', 'TemporalAnalyzer', 'MultimodalDetector']
