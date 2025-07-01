"""
File utilities for saving/loading results, images, and data
"""

import os
import datetime
from ..config.settings import *
import numpy as np
import cv2


def ensure_output_directory():
    """Ensure the output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_TRAJECTORIES_DIR, exist_ok=True)


def save_trajectories_to_file(trajectories, output_file=TRAJECTORIES_FILE):
    """Save full trajectories data to a file in the specified format:
    Track ID xmin ymin xmax ymax frame
    """
    ensure_output_directory()

    with open(output_file, 'w') as f:
        # Sort trajectories by track ID for consistent output
        for track_id in sorted(trajectories.keys()):
            trajectory = trajectories[track_id]
            for frame_data in trajectory:
                # Extract coordinates
                x1 = frame_data['position']['x1']
                y1 = frame_data['position']['y1']
                x2 = frame_data['position']['x2']
                y2 = frame_data['position']['y2']
                frame = frame_data['frame_number']

                # Write line in format: track_id xmin ymin xmax ymax frame
                f.write(f"{track_id} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {frame}\n")

    print(f"Trajectories saved to {output_file}")


def save_detailed_metrics_analysis(output_lines, output_file=DETAILED_METRICS_FILE):
    """
    Save detailed metrics analysis to a file.

    Parameters:
    -----------
    output_lines : list
        List of strings containing the detailed analysis output
    output_file : str
        Path to save the detailed metrics file
    """
    ensure_output_directory()

    try:
        with open(output_file, "w") as f:
            f.write("\n".join(output_lines))
        print(f"✅ Detailed metrics analysis saved to '{output_file}'")
    except Exception as e:
        print(f"⚠️ Warning: Could not save detailed metrics to file: {e}")


def save_final_analysis_results(accuracy_metrics, video_path, reference_path, annotations_path,
                               frame_width, frame_height, original_width, original_height,
                               frame_count, occlusion_enabled, kalman_enabled, output_file=FINAL_ANALYSIS_FILE):
    """
    Save final analysis results to a file.

    Parameters:
    -----------
    accuracy_metrics : dict
        Dictionary containing accuracy metrics
    video_path : str
        Path to the input video file
    reference_path : str
        Path to the reference image
    annotations_path : str
        Path to the annotations file
    frame_width : int
        Video frame width
    frame_height : int
        Video frame height
    original_width : int
        Original ground truth width
    original_height : int
        Original ground truth height
    frame_count : int
        Total number of frames processed
    occlusion_enabled : bool
        Whether occlusion tracking was enabled
    kalman_enabled : bool
        Whether Kalman filtering was enabled
    output_file : str
        Path to save the final analysis results file
    """
    ensure_output_directory()

    try:
        with open(output_file, "w") as f:
            f.write("=== FINAL ANALYSIS RESULTS ===\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video Path: {video_path}\n")
            f.write(f"Reference Path: {reference_path}\n")
            f.write(f"Annotations Path: {annotations_path}\n")
            f.write(f"Video Resolution: {frame_width}x{frame_height}\n")
            f.write(f"Original Resolution: {original_width}x{original_height}\n")
            f.write(f"Total Frames Processed: {frame_count}\n")
            f.write(f"Occlusion Tracking: {'Enabled' if occlusion_enabled else 'Disabled'}\n")
            f.write(f"Kalman Filtering: {'Enabled' if kalman_enabled else 'Disabled'}\n")
            f.write("\n")
            f.write("=== ACCURACY METRICS ===\n")
            f.write(f"Total ground truth objects: {accuracy_metrics['total_objects']}\n")
            f.write(f"Successfully tracked objects: {accuracy_metrics['matched_objects']}\n")
            f.write(f"Tracking success rate: {accuracy_metrics['matched_objects']/accuracy_metrics['total_objects']*100:.1f}%\n")
            f.write(f"Average IoU: {accuracy_metrics['average_iou']:.3f}\n")
            f.write(f"Average MSE: {accuracy_metrics['average_mse']:.3f}\n")

            # Add well-matched metrics if available
            if 'well_matched_objects' in accuracy_metrics:
                f.write(f"Very well matched objects: {accuracy_metrics['well_matched_objects']}\n")
                f.write(f"Average MSE (well matched): {accuracy_metrics.get('average_mse_well_matched', 0):.3f}\n")
                f.write(f"Average IoU (well matched): {accuracy_metrics.get('average_iou_well_matched', 0):.3f}\n")

            f.write("\n")
            f.write("=== CONFIGURATION ===\n")
            f.write(f"Background History: {BG_HISTORY}\n")
            f.write(f"Background Var Threshold: {BG_VAR_THRESHOLD}\n")
            f.write(f"Min Area: {MIN_AREA}\n")
            f.write(f"Max Area: {MAX_AREA}\n")
            f.write(f"Max Disappeared: {MAX_DISAPPEARED}\n")
            f.write(f"Max Distance: {MAX_DISTANCE}\n")
            f.write(f"Kalman Process Noise: {KALMAN_PROCESS_NOISE}\n")
            f.write(f"Kalman Measurement Noise: {KALMAN_MEASUREMENT_NOISE}\n")
            f.write(f"MSE Threshold: {MSE_THRESHOLD}\n")
            f.write(f"IoU Threshold: {IOU_THRESHOLD}\n")

        print(f"✅ Final analysis results saved to '{output_file}'")
    except Exception as e:
        print(f"⚠️ Warning: Could not save final analysis results to file: {e}")


def save_path_analysis_results(trajectory_stats, path_patterns, detailed_patterns, output_file=PATH_ANALYSIS_FILE):
    """
    Save path analysis results to a file.

    Parameters:
    -----------
    trajectory_stats : dict
        Dictionary containing trajectory statistics
    path_patterns : dict
        Dictionary containing path patterns and their frequencies
    detailed_patterns : dict
        Dictionary containing detailed pattern information
    output_file : str
        Path to save the path analysis results file
    """
    ensure_output_directory()

    try:
        with open(output_file, "w") as f:
            f.write("=== TRAJECTORY AND PATH PATTERN ANALYSIS RESULTS ===\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write path pattern statistics
            f.write("=== PATH PATTERN STATISTICS ===\n")
            f.write(f"Total path patterns found: {len(path_patterns)}\n\n")

            sorted_patterns = sorted(path_patterns.items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_patterns:
                f.write(f"{pattern}: {count} objects\n")
                f.write("Example objects:\n")
                for obj_info in detailed_patterns[pattern]:
                    f.write(f"  - Object {obj_info['object_id']} ({obj_info['class']})\n")
                    # if 'direction' in obj_info:
                    #     f.write(f"    Direction: {obj_info['direction']:.1f}°\n")
                    # if 'assigned' in obj_info:
                    #     f.write(f"    (Assigned to this path)\n")
                f.write("\n")

            # Write trajectory statistics
            f.write("=== TRAJECTORY STATISTICS ===\n")
            f.write(f"Total objects analyzed: {len(trajectory_stats)}\n\n")

            for obj_id, stats in trajectory_stats.items():
                f.write(f"Object {obj_id} ({stats['class']}):\n")
                f.write(f"  Duration: {stats['duration']} frames\n")
                f.write(f"  Total distance: {stats['total_distance']:.2f} pixels\n")
                f.write(f"  Average speed: {stats['avg_speed']:.2f} pixels/frame\n")
                f.write(f"  Start position: ({stats['start_pos'][0]:.1f}, {stats['start_pos'][1]:.1f})\n")
                f.write(f"  End position: ({stats['end_pos'][0]:.1f}, {stats['end_pos'][1]:.1f})\n")
                f.write(f"  Direction of movement: {stats['direction']:.1f}°\n\n")


        print(f"✅ Path analysis results saved to '{output_file}'")
    except Exception as e:
        print(f"⚠️ Warning: Could not save path analysis results to file: {e}")


def compute_occlusion_mask(with_occlusion_path, without_occlusion_path, save_path=OCCLUSION_MASK_IMAGE):
    """
    Compute occlusion mask by comparing two images.

    Parameters:
    -----------
    with_occlusion_path : str
        Path to image with occlusion
    without_occlusion_path : str
        Path to image without occlusion
    save_path : str
        Path to save the computed occlusion mask

    Returns:
    --------
    numpy.ndarray or None
        The computed occlusion mask
    """
    # Load images
    img_with = cv2.imread(with_occlusion_path, cv2.IMREAD_GRAYSCALE)
    img_without = cv2.imread(without_occlusion_path, cv2.IMREAD_GRAYSCALE)
    if img_with is None or img_without is None:
        print("Error: Could not load one or both images.")
        return None

    # Compute absolute difference
    diff = cv2.absdiff(img_with, img_without)

    # Threshold the difference to get the occlusion mask
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)  # 30 is a typical threshold, adjust if needed

    # Optional: Clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Save the mask
    cv2.imwrite(save_path, mask)
    print(f"Occlusion mask saved to {save_path}")

    return mask


def save_ground_truth_trajectories(last_frame, ground_truth_trajectories, output_dir, frame_width, frame_height, original_width, original_height):
    """Save ground truth trajectory images for each object"""
    if last_frame is None:
        print("No frame available for drawing ground truth trajectories")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each ground truth trajectory
    for object_id, trajectory in ground_truth_trajectories.items():
        if len(trajectory) < 2:
            continue

        # Create a copy of the last frame
        trajectory_image = last_frame.copy()

        # Get color for this object (use consistent color scheme)
        color = get_object_color(object_id)

        # Extract center points from ground truth trajectory
        x_coords = []
        y_coords = []
        for frame_data in trajectory:
            # Scale ground truth coordinates to video resolution
            x_scale = frame_width / original_width
            y_scale = frame_height / original_height

            x1 = frame_data['position']['x1'] * x_scale
            y1 = frame_data['position']['y1'] * y_scale
            x2 = frame_data['position']['x2'] * x_scale
            y2 = frame_data['position']['y2'] * y_scale

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x_coords.append(center_x)
            y_coords.append(center_y)

        # Convert to numpy array for drawing
        pts = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)

        # Draw the ground truth trajectory
        if pts.shape[0] > 1:
            cv2.polylines(trajectory_image, [pts], False, color, 3)

        # Draw start and end points
        start_point = (int(x_coords[0]), int(y_coords[0]))
        end_point = (int(x_coords[-1]), int(y_coords[-1]))

        # Draw start point as a circle
        cv2.circle(trajectory_image, start_point, 8, color, -1)
        cv2.circle(trajectory_image, start_point, 8, (255, 255, 255), 2)

        # Draw end point as a square
        square_size = 8
        cv2.rectangle(trajectory_image,
                     (end_point[0]-square_size, end_point[1]-square_size),
                     (end_point[0]+square_size, end_point[1]+square_size),
                     color, -1)
        cv2.rectangle(trajectory_image,
                     (end_point[0]-square_size, end_point[1]-square_size),
                     (end_point[0]+square_size, end_point[1]+square_size),
                     (255, 255, 255), 2)

        # Add object ID and trajectory information
        cv2.putText(trajectory_image, f"Ground Truth ID: {object_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(trajectory_image, f"Trajectory Points: {len(trajectory)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(trajectory_image, "GROUND TRUTH", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the image
        output_path = os.path.join(output_dir, f"ground_truth_trajectory_{object_id}.jpg")
        cv2.imwrite(output_path, trajectory_image)
        print(f"Saved ground truth trajectory image for object {object_id} to {output_path}")


def get_object_color(object_id):
    """Get a consistent color for an object ID"""
    # Use a simple hash-based color generation for consistency
    import random
    random.seed(object_id)  # Ensure consistent colors for same object ID
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    # Ensure at least one component is bright
    if max(b, g, r) < 200:
        r = 255

    return (b, g, r)