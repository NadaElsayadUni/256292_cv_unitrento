# Human Motion Analysis Package

# Core components
from .core.analyzer import HumanMotionAnalyzer
from .core.detection import ObjectDetector
from .core.tracking import ObjectTracker
from .core.kalman_filter import KalmanFilterManager
from .core.occlusion import OcclusionHandler

# Utility functions
from .utils.metrics import compute_trajectory_accuracy, calculate_mse, calculate_iou
from .utils.visualization import Visualizer
from .utils.file_utils import (
    save_trajectories_to_file,
    save_detailed_metrics_analysis,
    save_final_analysis_results,
    save_path_analysis_results
)

# Data handling
from .data.annotations import (
    read_annotations_file,
    analyze_trajectories,
    analyze_path_patterns,
    visualize_path_patterns,
    visualize_trajectories
)

# Configuration
from .config.settings import *

__all__ = [
    # Core classes
    'HumanMotionAnalyzer',
    'ObjectDetector',
    'ObjectTracker',
    'KalmanFilterManager',
    'OcclusionHandler',

    # Utility functions
    'compute_trajectory_accuracy',
    'calculate_mse',
    'calculate_iou',
    'Visualizer',
    'save_trajectories_to_file',
    'save_detailed_metrics_analysis',
    'save_final_analysis_results',
    'save_path_analysis_results',

    # Data functions
    'read_annotations_file',
    'analyze_trajectories',
    'analyze_path_patterns',
    'visualize_path_patterns',
    'visualize_trajectories',
]