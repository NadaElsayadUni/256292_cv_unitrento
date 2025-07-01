# Human Motion Analysis Package

A comprehensive Python package for analyzing human motion in videos using computer vision techniques. This package provides modular, maintainable code for object detection, tracking, Kalman filtering, occlusion handling, and trajectory analysis.

## 🎯 Features

- **Object Detection**: Background subtraction-based moving object detection
- **Object Tracking**: Multi-object tracking with consistent ID assignment
- **Kalman Filtering**: Smooth trajectory prediction and filtering
- **Occlusion Handling**: Advanced occlusion detection and object re-identification
- **Trajectory Analysis**: Complete trajectory reconstruction and analysis
- **Path Pattern Analysis**: Analysis of movement patterns and entry/exit points
- **Accuracy Metrics**: IoU and MSE-based accuracy evaluation against ground truth
- **Visualization**: Real-time and post-processing visualization tools
- **Modular Architecture**: Clean separation of concerns with reusable components

## 🚀 How to Run

### Prerequisites

1. **Python 3.6+** installed
2. **Required packages** (you already have these!):
   ```bash
   pip install opencv-python==4.9.0 opencv-contrib-python numpy open3d matplotlib scikit-image ipykernel pathlib2 networkx
   ```

   Or install from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start (Recommended)

1. **Navigate to the project directory**:
   ```bash
   cd project
   ```

2. **Run the simple run script**:
   ```bash
   python run.py
   ```

3. **Choose an option**:
   - `1` - Simple analysis (no occlusion tracking)
   - `2` - Analysis with occlusion tracking (recommended)
   - `3` - Trajectory and path pattern analysis
   - `4` - Show help

### Command Line Options

```bash
# Run simple analysis
python run.py 1
python run.py simple

# Run with occlusion tracking (recommended)
python run.py 2
python run.py occlusion

# Run trajectory and path pattern analysis
python run.py 3
python run.py path

# Show help
python run.py help
```

### Alternative: Direct Path Analysis

```bash
python run_path_analysis.py
```

This runs only the trajectory and path pattern analysis on ground truth data.

### What You'll See

- **Real-time visualization**: Object tracking with bounding boxes, IDs, and trajectories
- **Console output**: Progress updates and final accuracy metrics
- **Generated files**: Trajectory data and visualizations saved to disk

### Expected Output Files

After running, you'll find all output files in the `output/` folder:

- `output/trajectories.txt` - Raw trajectory data
- `output/trajectories_raw.jpg` - Raw trajectory visualization
- `output/trajectories_kalman.jpg` - Kalman-filtered visualization
- `output/last_frame.jpg` - Last processed frame
- `output/aligned_occlusion_mask.png` - Computed occlusion mask (if using occlusion tracking)
- `output/detailed_metrics_analysis.txt` - Detailed metrics analysis with per-object comparisons
- `output/final_analysis_results.txt` - Final analysis summary with configuration and results
- `output/path_patterns_graph.png` - Path patterns visualization (if running path analysis)
- `output/path_analysis_results.txt` - Path analysis results (if running path analysis)
- `output/detected_trajectories/` - Individual trajectory images for each tracked object
- `output/ground_truth_trajectories/` - Ground truth trajectory images for comparison

### Troubleshooting

**File not found errors**: Ensure you're in the `project` directory and the `Videos and Annotations` folder exists with the required files.

**Import errors**: Make sure you've installed the required packages with pip.

**Visualization issues**: The script will show real-time windows - press 'q' to quit the video processing.

## 📁 Project Structure

```
project/
├── src/                           # Main package source code
│   ├── core/                      # Core analysis components
│   │   ├── analyzer.py           # Main orchestrator class
│   │   ├── detection.py          # Object detection module
│   │   ├── tracking.py           # Object tracking module
│   │   ├── kalman_filter.py      # Kalman filtering module
│   │   └── occlusion.py          # Occlusion handling module
│   ├── utils/                     # Utility functions
│   │   ├── metrics.py            # Accuracy metrics calculation
│   │   ├── visualization.py      # Visualization utilities
│   │   └── file_utils.py         # File I/O operations
│   ├── data/                      # Data handling
│   │   └── annotations.py        # Annotation file reading
│   ├── config/                    # Configuration
│   │   └── settings.py           # Centralized settings
│   └── __init__.py               # Package exports
├── Videos and Annotations/        # Dataset directory
│   ├── video0/                   # Video 0 data
│   │   ├── reference.jpeg        # Background reference
│   │   ├── video.mp4             # Original video
│   │   ├── annotations.txt       # Ground truth annotations
│   │   └── masked/               # Masked video data
│   └── video3/                   # Video 3 data
├── output/                        # Output directory (created automatically)
│   ├── trajectories.txt          # Raw trajectory data
│   ├── trajectories_raw.jpg      # Raw trajectory visualization
│   ├── trajectories_kalman.jpg   # Kalman-filtered visualization
│   ├── last_frame.jpg            # Last processed frame
│   ├── aligned_occlusion_mask.png # Computed occlusion mask
│   ├── detailed_metrics_analysis.txt # Detailed metrics analysis
│   ├── final_analysis_results.txt # Final analysis summary
│   ├── path_patterns_graph.png   # Path patterns visualization
│   ├── path_analysis_results.txt # Path analysis results
│   ├── detected_trajectories/    # Individual trajectory images
│   └── ground_truth_trajectories/ # Ground truth trajectory images
├── example_usage.py              # Usage examples
├── run.py                        # Main run script
├── run_path_analysis.py          # Path analysis script
├── final_one.py                  # Original monolithic implementation
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy matplotlib pathlib
   ```

### Basic Usage

```python
from src import HumanMotionAnalyzer

# Initialize analyzer
analyzer = HumanMotionAnalyzer(
    video_path="Videos and Annotations/video0/video.mp4",
    reference_path="Videos and Annotations/video0/reference.jpeg",
    annotations_path="Videos and Annotations/video0/annotations.txt"
)

# Run analysis
accuracy_metrics = analyzer.run_analysis(
    show_visualization=True,
    save_results=True
)

# View results
print(f"Tracking success rate: {accuracy_metrics['matched_objects']/accuracy_metrics['total_objects']*100:.1f}%")
```

### Advanced Usage with Occlusion Tracking

```python
# Run analysis with occlusion tracking
accuracy_metrics = analyzer.run_analysis_with_occlusion_tracking(
    with_occlusion_path="Videos and Annotations/video0/reference.jpeg",
    without_occlusion_path="Videos and Annotations/video0/masked/masked_reference_0.jpeg",
    show_visualization=True,
    save_results=True
)
```

## 📖 Detailed Usage Examples

### 1. Simple Analysis (No Occlusion Tracking)

```python
from src import HumanMotionAnalyzer
from pathlib import Path

# Setup paths
base_path = Path("Videos and Annotations")
video_path = base_path / "video0" / "video.mp4"
reference_path = base_path / "video0" / "reference.jpeg"
annotations_path = base_path / "video0" / "annotations.txt"

# Initialize and run
analyzer = HumanMotionAnalyzer(
    video_path=str(video_path),
    reference_path=str(reference_path),
    annotations_path=str(annotations_path)
)

accuracy_metrics = analyzer.run_analysis()
```

### 2. Analysis with Occlusion Tracking

```python
# Use masked video data for occlusion tracking
video_path = base_path / "video0" / "masked" / "masked_video_0.mp4"
masked_reference_path = base_path / "video0" / "masked" / "masked_reference_0.jpeg"

analyzer = HumanMotionAnalyzer(
    video_path=str(video_path),
    reference_path=str(reference_path),
    annotations_path=str(annotations_path)
)

# Run with occlusion tracking
accuracy_metrics = analyzer.run_analysis_with_occlusion_tracking(
    with_occlusion_path=str(reference_path),
    without_occlusion_path=str(masked_reference_path)
)
```

### 3. Individual Component Usage

```python
from src import ObjectDetector, ObjectTracker, KalmanFilterManager, OcclusionHandler
from src import read_annotations_file, calculate_mse, calculate_iou

# Use individual components
detector = ObjectDetector()
tracker = ObjectTracker(frame_width=1920, frame_height=1080)
kalman_manager = KalmanFilterManager()
occlusion_handler = OcclusionHandler()

# Read annotations
trajectories, class_counts, frame_distribution = read_annotations_file("annotations.txt")
```

### 4. Trajectory and Path Pattern Analysis

```python
from src.data.annotations import (
    read_annotations_file,
    analyze_trajectories,
    analyze_path_patterns,
    visualize_path_patterns
)
from src.utils.file_utils import save_path_analysis_results

# Read ground truth trajectories
trajectories, class_counts, frame_distribution = read_annotations_file("annotations.txt")

# Analyze trajectories
trajectory_stats = analyze_trajectories(trajectories, class_counts, frame_distribution)

# Analyze path patterns
path_patterns, detailed_patterns = analyze_path_patterns(
    trajectories, "reference.jpeg", trajectory_stats
)

# Create visualization
visualize_path_patterns(path_patterns, "path_patterns_graph.png")

# Save results
save_path_analysis_results(trajectory_stats, path_patterns, detailed_patterns)
```

### 5. Manual Occlusion Mask Computation

```python
from src.core.occlusion import OcclusionHandler

# Create occlusion handler and compute mask
occlusion_handler = OcclusionHandler()
mask = occlusion_handler.compute_occlusion_mask(
    with_occlusion_path="reference.jpeg",
    without_occlusion_path="masked_reference.jpeg",
    save_path="occlusion_mask.png"
)

# Use with analyzer
analyzer = HumanMotionAnalyzer(...)
analyzer.set_occlusion_mask(mask)
```

## ⚙️ Configuration

All parameters are centralized in `src/config/settings.py`:

```python
# Background subtraction parameters
BG_HISTORY = 500
BG_VAR_THRESHOLD = 20
BG_DETECT_SHADOWS = False

# Object detection parameters
MIN_AREA = 600
MAX_AREA = 10000

# Tracking parameters
MAX_DISAPPEARED = 150
MAX_BOUNDARY_DISAPPEARED = 5
MAX_DISTANCE = 100

# Kalman filter parameters
KALMAN_PROCESS_NOISE = 0.1
KALMAN_MEASUREMENT_NOISE = 0.005

# Occlusion parameters
OCCLUSION_PROXIMITY_DISTANCE = 20
OCCLUSION_STATIONARY_THRESHOLD = 10
OCCLUSION_STATIONARY_FRAMES = 5

# Accuracy metrics thresholds
MSE_THRESHOLD = 15
IOU_THRESHOLD = 0.1
MIN_COMMON_FRAMES = 5
```

## 📊 Output Files

The analysis generates several output files in the `output/` folder:

- **`output/trajectories.txt`**: Raw trajectory data in format `track_id xmin ymin xmax ymax frame`
- **`output/trajectories_raw.jpg`**: Visualization of raw trajectories
- **`output/trajectories_kalman.jpg`**: Visualization of Kalman-filtered trajectories
- **`output/last_frame.jpg`**: Last processed frame
- **`output/aligned_occlusion_mask.png`**: Computed occlusion mask aligned to video dimensions
- **`output/detailed_metrics_analysis.txt`**: Detailed metrics analysis with per-object comparisons
- **`output/final_analysis_results.txt`**: Final analysis summary with configuration and results
- **`output/path_patterns_graph.png`**: Path patterns visualization (if running path analysis)
- **`output/path_analysis_results.txt`**: Path analysis results (if running path analysis)
- **`output/detected_trajectories/`**: Individual trajectory images for each tracked object
- **`output/ground_truth_trajectories/`**: Ground truth trajectory images for comparison

## 🔧 Core Components

### HumanMotionAnalyzer
Main orchestrator class that coordinates all analysis components.

**Key Methods:**
- `run_analysis()`: Basic analysis pipeline
- `run_analysis_with_occlusion_tracking()`: Analysis with occlusion handling
- `process_frame()`: Process single frame
- `compute_accuracy()`: Calculate accuracy metrics

### ObjectDetector
Handles moving object detection using background subtraction.

**Key Methods:**
- `detect_moving_objects(frame)`: Detect objects in a frame

### ObjectTracker
Manages object tracking, ID assignment, and trajectory management.

**Key Methods:**
- `track_objects(detections, frame, frame_count, occlusion_handler)`: Track objects
- `update_trajectories_with_kalman()`: Update trajectories with Kalman filtering

### KalmanFilterManager
Manages Kalman filters for multiple objects.

**Key Methods:**
- `update_kalman_filters(tracked_objects)`: Update all filters
- `get_kalman_center(object_id)`: Get filtered position
- `get_kalman_velocity(object_id)`: Get velocity estimate

### OcclusionHandler
Handles occlusion detection and object re-identification.

**Key Methods:**
- `compute_occlusion_mask()`: Compute occlusion mask from images
- `is_point_near_occlusion(x, y)`: Check if point is near occlusion
- `track_occluded_objects()`: Track objects during occlusion

## 📈 Performance Metrics

The package provides comprehensive accuracy metrics:

- **IoU (Intersection over Union)**: Measures bounding box overlap accuracy
- **MSE (Mean Squared Error)**: Measures position accuracy
- **Tracking Success Rate**: Percentage of successfully tracked objects
- **Frame Coverage**: Percentage of frames with successful tracking

## 🎨 Visualization

Real-time visualization includes:
- Object bounding boxes with consistent colors
- Object IDs and status indicators
- Trajectory paths
- Kalman filter predictions
- Occlusion area highlighting
- Velocity vectors

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the `project` directory
2. **File Not Found**: Check that all video and annotation files exist
3. **Memory Issues**: Reduce video resolution or use shorter clips
4. **Poor Detection**: Adjust `MIN_AREA` and `MAX_AREA` in settings

### Debug Mode

Enable debug output by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Example Scripts

Run the example usage script:
```bash
python example_usage.py
```

Choose from:
1. Simple usage (no occlusion tracking)
2. Analysis with occlusion tracking
3. Individual components usage
4. Manual occlusion mask computation

## 📄 License

This project is part of the Computer Vision course.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- NumPy for numerical computations
- Matplotlib for visualization

---

**Note**: This package is designed for educational and research purposes in human motion analysis and computer vision applications.