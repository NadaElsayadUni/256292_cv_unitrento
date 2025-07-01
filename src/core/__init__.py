"""
Core analysis package

Contains the main analysis components:
- Object detection and tracking
- Kalman filtering
- Occlusion handling
- Main analyzer orchestrator
"""

from .tracking import ObjectTracker
from .detection import ObjectDetector
from .kalman_filter import KalmanFilterManager
from .occlusion import OcclusionHandler
from .analyzer import HumanMotionAnalyzer

__all__ = [
    'ObjectTracker',
    'ObjectDetector',
    'KalmanFilterManager',
    'OcclusionHandler',
    'HumanMotionAnalyzer',
]