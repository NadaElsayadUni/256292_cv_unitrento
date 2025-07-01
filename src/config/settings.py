"""
Configuration settings for Human Motion Analysis
"""

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
MAX_TRAJECTORY_POINTS = 30

# Kalman filter parameters
KALMAN_PROCESS_NOISE = 0.1
KALMAN_MEASUREMENT_NOISE = 0.005
KALMAN_ERROR_COV = 0.1

# Occlusion parameters
OCCLUSION_PROXIMITY_DISTANCE = 20
OCCLUSION_STATIONARY_THRESHOLD = 10
OCCLUSION_STATIONARY_FRAMES = 5
OCCLUSION_APPEARED_COUNTER = 10

# Accuracy metrics thresholds
MSE_THRESHOLD = 15
IOU_THRESHOLD = 0.1
MIN_COMMON_FRAMES = 5

# Visualization parameters
TRAJECTORY_THICKNESS = 2
BOUNDING_BOX_THICKNESS = 2
CENTER_POINT_RADIUS = 4
VELOCITY_SCALE = 10
DASH_LENGTH = 8

# File paths
BASE_PATH = "Videos and Annotations"
OUTPUT_DIR = "output"
TRAJECTORIES_FILE = "output/trajectories.txt"
DETAILED_METRICS_FILE = "output/detailed_metrics_analysis.txt"
FINAL_ANALYSIS_FILE = "output/final_analysis_results.txt"
PATH_ANALYSIS_FILE = "output/path_analysis_results.txt"
PATH_PATTERNS_GRAPH = "output/path_patterns_graph.png"
TRAJECTORIES_RAW_IMAGE = "output/trajectories_raw.jpg"
TRAJECTORIES_KALMAN_IMAGE = "output/trajectories_kalman.jpg"
LAST_FRAME_IMAGE = "output/last_frame.jpg"
OCCLUSION_MASK_IMAGE = "output/aligned_occlusion_mask.png"
INDIVIDUAL_TRAJECTORIES_DIR = "output/detected_trajectories"
GROUND_TRUTH_TRAJECTORIES_DIR = "output/ground_truth_trajectories"