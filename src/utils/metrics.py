"""
Metrics utilities for accuracy and evaluation
"""

import numpy as np
from ..data.annotations import read_annotations_file
from ..utils.file_utils import save_detailed_metrics_analysis
from ..config.settings import *


def calculate_mse(gt_trajectory, detected_trajectory, frame_width, frame_height, original_width=1920, original_height=1080):
    """Calculate Mean Squared Error between two trajectories using bounding box coordinates"""
    # Create dictionaries for easy lookup by frame number
    gt_dict = {frame['frame_number']: frame for frame in gt_trajectory}
    detected_dict = {frame['frame_number']: frame for frame in detected_trajectory}

    # Find common frames
    common_frames = set(gt_dict.keys()) & set(detected_dict.keys())
    if not common_frames:
        return float('inf')

    # Need at least MIN_COMMON_FRAMES common frames to consider it a valid match
    if len(common_frames) < MIN_COMMON_FRAMES:
        return float('inf')

    # Calculate scaling factors to convert ground truth to video resolution
    x_scale = frame_width / original_width
    y_scale = frame_height / original_height

    total_error = 0
    for frame_num in common_frames:
        gt_frame = gt_dict[frame_num]
        detected_frame = detected_dict[frame_num]

        # Scale ground truth coordinates to match video resolution
        gt_x1 = gt_frame['position']['x1'] * x_scale
        gt_y1 = gt_frame['position']['y1'] * y_scale
        gt_x2 = gt_frame['position']['x2'] * x_scale
        gt_y2 = gt_frame['position']['y2'] * y_scale

        # Get detected coordinates
        det_x1 = detected_frame['position']['x1']
        det_y1 = detected_frame['position']['y1']
        det_x2 = detected_frame['position']['x2']
        det_y2 = detected_frame['position']['y2']

        # Calculate box sizes for normalization
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1
        det_width = det_x2 - det_x1
        det_height = det_y2 - det_y1

        # Calculate normalized error
        error = (
            ((gt_x1 - det_x1) / max(gt_width, det_width)) ** 2 +
            ((gt_y1 - det_y1) / max(gt_height, det_height)) ** 2 +
            ((gt_x2 - det_x2) / max(gt_width, det_width)) ** 2 +
            ((gt_y2 - det_y2) / max(gt_height, det_height)) ** 2
        )
        total_error += error

    # Return normalized MSE (average error per frame)
    avg_error = total_error / len(common_frames)
    # print(f"\nAverage MSE across {len(common_frames)} frames: {avg_error:.3f}")
    return avg_error


def calculate_iou(gt_trajectory, detected_trajectory, frame_width, frame_height, original_width=1920, original_height=1080):
    """Calculate Intersection over Union (IoU) between two trajectories"""
    # Create dictionaries for easy lookup by frame number
    gt_dict = {frame['frame_number']: frame for frame in gt_trajectory}
    detected_dict = {frame['frame_number']: frame for frame in detected_trajectory}

    # Find common frames
    common_frames = set(gt_dict.keys()) & set(detected_dict.keys())
    if not common_frames:
        return 0.0

    # Need at least MIN_COMMON_FRAMES common frames to consider it a valid match
    if len(common_frames) < MIN_COMMON_FRAMES:
        return 0.0

    # Calculate scaling factors to convert ground truth to video resolution
    x_scale = frame_width / original_width
    y_scale = frame_height / original_height

    total_iou = 0
    for frame_num in common_frames:
        gt_frame = gt_dict[frame_num]
        detected_frame = detected_dict[frame_num]

        # Scale ground truth coordinates to match video resolution
        gt_x1 = gt_frame['position']['x1'] * x_scale
        gt_y1 = gt_frame['position']['y1'] * y_scale
        gt_x2 = gt_frame['position']['x2'] * x_scale
        gt_y2 = gt_frame['position']['y2'] * y_scale

        # Get detected coordinates
        det_x1 = detected_frame['position']['x1']
        det_y1 = detected_frame['position']['y1']
        det_x2 = detected_frame['position']['x2']
        det_y2 = detected_frame['position']['y2']

        # Calculate intersection coordinates
        inter_x1 = max(gt_x1, det_x1)
        inter_y1 = max(gt_y1, det_y1)
        inter_x2 = min(gt_x2, det_x2)
        inter_y2 = min(gt_y2, det_y2)

        # Calculate areas
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = gt_area + det_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        total_iou += iou

    # Return average IoU
    avg_iou = total_iou / len(common_frames)
    # print(f"\nAverage IoU across {len(common_frames)} frames: {avg_iou:.3f}")
    return avg_iou


def compute_trajectory_accuracy(full_trajectories, annotations_path, frame_width, frame_height, original_width=1920, original_height=1080):
    """Compute accuracy metrics between detected and ground truth trajectories"""
    # Read ground truth trajectories from annotations file
    ground_truth_trajectories, _, _ = read_annotations_file(annotations_path)

    if not ground_truth_trajectories:
        print("Error: Could not read ground truth trajectories")
        return None

    print(f"\nDebug: Found {len(ground_truth_trajectories)} ground truth trajectories")
    print(f"Debug: Found {len(full_trajectories)} detected trajectories")

    # Prepare output for both console and file
    output_lines = []
    output_lines.append("=== TRAJECTORY ACCURACY ANALYSIS ===")
    output_lines.append(f"Debug: Found {len(ground_truth_trajectories)} ground truth trajectories")
    output_lines.append(f"Debug: Found {len(full_trajectories)} detected trajectories")
    output_lines.append("")

    # Dictionary to store accuracy metrics
    accuracy_metrics = {
        'mse': {},  # Mean Squared Error for each object
        'iou': {},  # Intersection over Union for each object
        'matched_objects': 0,  # Number of objects that were successfully tracked
        'well_matched_objects': 0,  # Number of objects that were well matched
        'total_objects': len(ground_truth_trajectories),
        'average_mse': 0.0,
        'average_iou': 0.0,
        'average_mse_well_matched': 0.0,
        'average_iou_well_matched': 0.0
    }

    # For each ground truth trajectory, find the best matching detected trajectory
    for gt_id, gt_trajectory in ground_truth_trajectories.items():
        best_iou = 0.0  # Start with 0 IoU
        best_match_id = None

        # First find the best match using IoU
        for detected_id, detected_trajectory in full_trajectories.items():
            # Calculate IoU for matching
            iou = calculate_iou(gt_trajectory, detected_trajectory, frame_width, frame_height, original_width, original_height)
            # comparison_msg = f"\nComparing ground truth {gt_id} with detected {detected_id}:"
            # iou_msg = f"IoU: {iou:.3f}"
            # print(comparison_msg)
            # print(iou_msg)
            # output_lines.append(comparison_msg)
            # output_lines.append(iou_msg)

            # Update best match if this IoU is better (no threshold)
            if iou > best_iou:
                best_iou = iou
                best_match_id = detected_id

        # If we found a good match, calculate MSE for that match
        if best_match_id is not None:
            # match_msg = f"\nFound best match for ground truth {gt_id}: detected {best_match_id}"
            # best_iou_msg = f"Best IoU: {best_iou:.3f}"
            # print(match_msg)
            # print(best_iou_msg)
            # output_lines.append(match_msg)
            # output_lines.append(best_iou_msg)

            # Calculate MSE for the best match
            mse = calculate_mse(gt_trajectory, full_trajectories[best_match_id], frame_width, frame_height, original_width, original_height)
            # mse_msg = f"MSE for best match: {mse:.3f}"
            # print(mse_msg)
            # output_lines.append(mse_msg)

            accuracy_metrics['mse'][gt_id] = {
                'detected_id': best_match_id,
                'mse': mse
            }
            accuracy_metrics['iou'][gt_id] = {
                'detected_id': best_match_id,
                'iou': best_iou
            }
            accuracy_metrics['matched_objects'] += 1
            if mse <= MSE_THRESHOLD and best_iou >= IOU_THRESHOLD:
                accuracy_metrics['well_matched_objects'] += 1
        else:
            no_match_msg = f"\nNo good match found for ground truth object {gt_id}"
            # print(no_match_msg)
            # output_lines.append(no_match_msg)

    # Calculate average MSE and IoU
    if accuracy_metrics['matched_objects'] > 0:
        accuracy_metrics['average_mse'] = sum(m['mse'] for m in accuracy_metrics['mse'].values()) / accuracy_metrics['matched_objects']
        accuracy_metrics['average_iou'] = sum(i['iou'] for i in accuracy_metrics['iou'].values()) / accuracy_metrics['matched_objects']

    if accuracy_metrics['well_matched_objects'] > 0:
        accuracy_metrics['average_mse_well_matched'] = sum(m['mse'] for m in accuracy_metrics['mse'].values() if m['mse'] <= MSE_THRESHOLD) / accuracy_metrics['well_matched_objects']
        accuracy_metrics['average_iou_well_matched'] = sum(i['iou'] for i in accuracy_metrics['iou'].values() if i['iou'] >= IOU_THRESHOLD) / accuracy_metrics['well_matched_objects']

    # Prepare final results
    final_results = []
    final_results.append("\n=== Final Accuracy Metrics ===")
    final_results.append(f"Total ground truth objects: {accuracy_metrics['total_objects']}")
    final_results.append(f"Successfully matched objects: {accuracy_metrics['matched_objects']}")
    final_results.append(f"Average IoU: {accuracy_metrics['average_iou']:.3f}")
    final_results.append(f"Average MSE: {accuracy_metrics['average_mse']:.3f}")
    final_results.append(f"Very Well matched objects: {accuracy_metrics['well_matched_objects']}")
    final_results.append(f"Average MSE well matched: {accuracy_metrics['average_mse_well_matched']:.3f}")
    final_results.append(f"Average IoU well matched: {accuracy_metrics['average_iou_well_matched']:.3f}")

    # Print final results
    for line in final_results:
        print(line)
        output_lines.append(line)

    # Detailed metrics per object
    detailed_metrics = []
    detailed_metrics.append("\nDetailed metrics per object:")
    for gt_id, metrics in accuracy_metrics['mse'].items():
        detected_id = metrics['detected_id']
        mse = metrics['mse']
        iou = accuracy_metrics['iou'][gt_id]['iou']
        detail_line = f"Ground truth ID {gt_id} -> Detected ID {detected_id}: MSE = {mse:.3f}, IoU = {iou:.3f}"
        detailed_metrics.append(detail_line)
        # print(detail_line)
        output_lines.append(detail_line)

    # Save detailed output to file
    save_detailed_metrics_analysis(output_lines)

    return accuracy_metrics