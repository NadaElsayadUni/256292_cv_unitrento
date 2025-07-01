"""
Data handling package

Contains functions for:
- Reading and parsing annotation files
- Data preprocessing and validation
- Trajectory analysis and visualization
- Path pattern analysis
"""

from .annotations import (
    read_annotations_file,
    generate_random_color,
    visualize_trajectories,
    analyze_trajectories,
    analyze_path_patterns,
    visualize_path_patterns
)

__all__ = [
    'read_annotations_file',
    'generate_random_color',
    'visualize_trajectories',
    'analyze_trajectories',
    'analyze_path_patterns',
    'visualize_path_patterns',
]