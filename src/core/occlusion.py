"""
Occlusion handling module for object tracking
"""

import cv2
import numpy as np
import os
from ..config.settings import *


class OcclusionHandler:
    """Handles occlusion detection and object tracking during occlusions"""

    def __init__(self):
        """Initialize the occlusion handler"""
        self.occlusion_mask = None
        self.occluded_objects = {}  # Dictionary to store objects in occlusion
        self.prev_positions = {}    # Store previous positions for velocity calculation

    def compute_occlusion_mask(self, with_occlusion_path, without_occlusion_path, save_path=OCCLUSION_MASK_IMAGE):
        """
        Compute occlusion mask from two reference images.

        This function compares two images (with and without occlusion) to create
        a binary mask indicating the occlusion area.

        Parameters:
        -----------
        with_occlusion_path : str
            Path to image with occlusion
        without_occlusion_path : str
            Path to image without occlusion
        save_path : str
            Path to save the computed mask

        Returns:
        --------
        numpy.ndarray or None
            The computed occlusion mask
        """
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        # Show and save the mask
        cv2.imshow("Occlusion Mask", mask)
        cv2.imwrite(save_path, mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return mask

    def set_occlusion_mask(self, mask):
        """Set the occlusion mask for the analyzer."""
        self.occlusion_mask = mask
        print(f"Occlusion mask set with shape: {mask.shape if mask is not None else 'None'}")

    def align_occlusion_mask(self, video_width, video_height):
        """Align the occlusion mask with the video dimensions."""
        if self.occlusion_mask is None:
            print("No occlusion mask to align")
            return

        mask_height, mask_width = self.occlusion_mask.shape
        print(f"Mask dimensions: {mask_width}x{mask_height}")
        print(f"Video dimensions: {video_width}x{video_height}")

        # Resize mask to match video dimensions
        if mask_width != video_width or mask_height != video_height:
            print(f"Resizing occlusion mask from {mask_width}x{mask_height} to {video_width}x{video_height}")
            self.occlusion_mask = cv2.resize(self.occlusion_mask, (video_width, video_height))

        # Save the aligned mask for inspection
        cv2.imwrite(OCCLUSION_MASK_IMAGE, self.occlusion_mask)
        print(f"Aligned occlusion mask saved as '{OCCLUSION_MASK_IMAGE}'")

    def is_point_in_occlusion(self, x, y):
        """Check if a point (x, y) is inside the occlusion mask."""
        if self.occlusion_mask is None:
            return False
        if 0 <= y < self.occlusion_mask.shape[0] and 0 <= x < self.occlusion_mask.shape[1]:
            return self.occlusion_mask[int(y), int(x)] > 0
        return False

    def is_point_near_occlusion(self, x, y, proximity_distance=50):
        """Check if a point (x, y) is near the occlusion mask area."""
        if self.occlusion_mask is None:
            return False

        # Check if point is within proximity_distance pixels of the occlusion area
        y_coords, x_coords = np.where(self.occlusion_mask > 0)
        if len(y_coords) == 0:
            return False

        # Calculate distances to all occlusion pixels
        distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        min_distance = np.min(distances)

        return min_distance <= proximity_distance

    def check_occlusion_entry(self, object_id, center, frame_count):
        """Check if an object has entered the occlusion area."""
        x, y = center

        # Check if point is in occlusion
        if self.is_point_in_occlusion(x, y):
            if object_id not in self.occluded_objects:
                print(f"Object {object_id} entered occlusion at position ({x}, {y})")
                return True
        return False

    def visualize_occlusion_mask(self, frame):
        """Visualize the occlusion mask overlay on a frame for debugging."""
        if self.occlusion_mask is None:
            return frame

        # Create a colored overlay
        overlay = frame.copy()

        # Create a colored mask (red for occlusion area)
        colored_mask = np.zeros_like(frame)
        colored_mask[self.occlusion_mask > 0] = [0, 0, 255]  # Red color

        # Blend the mask with the frame
        alpha = 0.3  # Transparency
        overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)

        # Add text
        cv2.putText(overlay, "Occlusion Area (Red)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return overlay

    def track_occluded_objects(self, new_detected_occluded_object, ground_truth_trajectories, full_trajectories, frame_width, frame_height, original_width, original_height):
        """Track occluded objects with improved matching strategies."""
        best_match_id = None
        best_match_score = 0.0
        best_gt_id = None
        best_gt_trajectory = None

        # Strategy ground truth matching
        for object_id, obj in self.occluded_objects.items():
            matched_gt_id, matched_gt_trajectory, match_score = self.match_occluded_object_to_ground_truth(
                object_id, obj, new_detected_occluded_object, ground_truth_trajectories,
                full_trajectories, frame_width, frame_height, original_width, original_height
            )
            if matched_gt_id is not None:
                print(f"✅ SUCCESS: Object {object_id} matched to ground truth {matched_gt_id}")
                print(f"   Match score: {match_score:.3f}")
                print(f"   Ground truth trajectory has {len(matched_gt_trajectory)} frames")
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_id = object_id
                    best_gt_id = matched_gt_id
                    best_gt_trajectory = matched_gt_trajectory

        # Set minimum threshold for accepting a match
        min_score_threshold = 0.1
        if best_match_id is not None and best_match_score >= min_score_threshold:
            print(f"\n=== MATCH FOUND ===")
            print(f"Best match: Object {best_match_id} with score {best_match_score:.3f}")
            return best_match_id, best_gt_id, best_gt_trajectory, best_match_score
        else:
            print(f"\n=== NO MATCH FOUND ===")
            if best_match_id is not None:
                print(f"Best candidate was Object {best_match_id} with score {best_match_score:.3f} (below threshold {min_score_threshold})")
            else:
                print("No candidates found")
            return None, None, None, 0.0

    def match_occluded_object_to_ground_truth(self, occluded_object_id, occluded_object_data, new_detection_data,
                                            ground_truth_trajectories, full_trajectories, frame_width, frame_height,
                                            original_width, original_height):
        """
        Match an occluded object with ground truth trajectories using IoU comparison.
        """
        print(f"\n=== Matching occluded object {occluded_object_id} to ground truth ===")

        # Get the last known position and frame of the occluded object
        last_frame = occluded_object_data['last_near_occlusion_frame']
        last_bbox = occluded_object_data['last_near_occlusion_bbox']
        last_center = occluded_object_data['last_visible_center']

        # Get new detection data if available
        if new_detection_data:
            new_frame = new_detection_data['first_near_occlusion_frame']
            new_bbox = new_detection_data['bbox']
            new_center = new_detection_data['center']

        best_match_gt_id = None
        best_match_score = 0.0
        best_match_trajectory = None

        # Check if we have ground truth trajectories
        if not ground_truth_trajectories:
            print("No ground truth trajectories available for matching")
            return None, None, 0.0

        # Get the detected object's trajectory before occlusion
        detected_trajectory = full_trajectories.get(occluded_object_id, [])
        if not detected_trajectory:
            print(f"No trajectory data found for object {occluded_object_id}")
            return None, None, 0.0

        # Iterate through all ground truth trajectories
        for gt_id, gt_trajectory in ground_truth_trajectories.items():
            print(f"\nChecking ground truth trajectory {gt_id}")

            # Find frames in ground truth that are before the occlusion frame
            gt_frames_before_occlusion = []
            for frame_data in gt_trajectory:
                if frame_data['frame_number'] <= last_frame:
                    gt_frames_before_occlusion.append(frame_data)

            if not gt_frames_before_occlusion:
                print(f"  No frames before occlusion in GT {gt_id}")
                continue

            # print(f"  GT {gt_id} has {len(gt_frames_before_occlusion)} frames before occlusion")

            # Check 1: Ensure both trajectories have common frames in the same time period
            # Create dictionaries for efficient O(1) lookup
            detected_dict = {frame['frame_number']: frame for frame in detected_trajectory}
            gt_dict = {frame['frame_number']: frame for frame in gt_frames_before_occlusion}

            # Find common frame numbers using set intersection
            common_frame_numbers = set(detected_dict.keys()) & set(gt_dict.keys())

            if len(common_frame_numbers) < MIN_COMMON_FRAMES:  # Need at least 5 common frames
                print(f"  Insufficient common frames: {len(common_frame_numbers)} (need at least {MIN_COMMON_FRAMES})")
                continue

            # Get the actual frame data pairs
            common_frames = [(gt_dict[frame_num], detected_dict[frame_num])
                            for frame_num in sorted(common_frame_numbers)]

            print(f"  Found {len(common_frames)} common frames between detected and GT {gt_id}")
            # print(f"  Common frame numbers: {sorted(common_frame_numbers)}")

            # Check 2: Calculate trajectory similarity using common frames
            total_iou = 0.0
            for gt_frame, detected_frame in common_frames:
                gt_bbox = gt_frame['position']
                det_bbox = detected_frame['position']
                iou = self.compute_iou(gt_bbox, det_bbox, frame_width, frame_height, original_width, original_height)
                total_iou += iou
                print(f"    Frame {gt_frame['frame_number']}: IoU = {iou:.3f}")

            # Calculate average IoU across common frames
            avg_iou = total_iou / len(common_frames)
            # print(f"  Average IoU across {len(common_frames)} common frames: {avg_iou:.3f}")

            # Check 3: If we have new detection data, verify it matches the ground truth after occlusion
            new_detection_score = 0.0
            if new_detection_data:
                # Find frames in ground truth after the occlusion period
                gt_frames_after_occlusion = []
                for frame_data in gt_trajectory:
                    if frame_data['frame_number'] >= new_frame:
                        gt_frames_after_occlusion.append(frame_data)

                if gt_frames_after_occlusion:
                    # Find the closest frame to the new detection
                    closest_frame = min(gt_frames_after_occlusion,
                                      key=lambda x: abs(x['frame_number'] - new_frame))

                    # Calculate IoU between new detection and ground truth
                    gt_bbox_after = closest_frame['position']
                    iou_after = self.compute_iou(gt_bbox_after, new_bbox, frame_width, frame_height, original_width, original_height)
                    new_detection_score = iou_after

                    print(f"  New detection IoU with GT {gt_id} at frame {closest_frame['frame_number']}: {iou_after:.3f}")
                else:
                    print(f"  No frames after occlusion in GT {gt_id}")

            # Calculate frame coverage (how many frames we could match)
            frame_coverage = len(common_frames) / len(gt_frames_before_occlusion)
            print(f"  Frame coverage: {frame_coverage:.3f} ({len(common_frames)}/{len(gt_frames_before_occlusion)})")

            # Combined score: IoU + coverage + new detection verification
            if new_detection_data:
                combined_score = (avg_iou * 0.5) + (frame_coverage * 0.2) + (new_detection_score * 0.3)
                print(f"  Combined score: {combined_score:.3f} (IoU: {avg_iou:.3f}, Coverage: {frame_coverage:.3f}, New: {new_detection_score:.3f})")
            else:
                combined_score = (avg_iou * 0.7) + (frame_coverage * 0.3)
                print(f"  Combined score: {combined_score:.3f} (IoU: {avg_iou:.3f}, Coverage: {frame_coverage:.3f})")

            # Update best match if this score is better
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_match_gt_id = gt_id
                best_match_trajectory = gt_trajectory
                print(f"  *** New best match: GT {gt_id} with score {combined_score:.3f} ***")

        # Set minimum threshold for accepting a match
        min_score_threshold = 0.1
        if best_match_gt_id is not None and best_match_score >= min_score_threshold:
            print(f"\n=== MATCH FOUND ===")
            print(f"Occluded object {occluded_object_id} matched to ground truth {best_match_gt_id}")
            print(f"Match score: {best_match_score:.3f}")
            return best_match_gt_id, best_match_trajectory, best_match_score
        else:
            print(f"\n=== NO MATCH FOUND ===")
            if best_match_gt_id is not None:
                print(f"Best candidate was GT {best_match_gt_id} with score {best_match_score:.3f} (below threshold {min_score_threshold})")
            else:
                print("No candidates found")
            return None, None, 0.0

    def compute_iou(self, gt_bbox, detected_bbox, frame_width, frame_height, original_width, original_height):
        """Compute IoU between ground truth and detected bounding boxes"""
        # Calculate scaling factors to convert ground truth to video resolution
        x_scale = frame_width / original_width
        y_scale = frame_height / original_height

        # Scale ground truth coordinates to match video resolution
        gt_x1 = gt_bbox['x1'] * x_scale
        gt_y1 = gt_bbox['y1'] * y_scale
        gt_x2 = gt_bbox['x2'] * x_scale
        gt_y2 = gt_bbox['y2'] * y_scale

        # Get detected coordinates - handle both tuple and dictionary formats
        if isinstance(detected_bbox, tuple):
            # Tuple format: (x, y, width, height)
            det_x1 = detected_bbox[0]  # x
            det_y1 = detected_bbox[1]  # y
            det_x2 = detected_bbox[0] + detected_bbox[2]  # x + width
            det_y2 = detected_bbox[1] + detected_bbox[3]  # y + height
        elif isinstance(detected_bbox, dict):
            # Dictionary format: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            det_x1 = detected_bbox['x1']
            det_y1 = detected_bbox['y1']
            det_x2 = detected_bbox['x2']
            det_y2 = detected_bbox['y2']
        else:
            print(f"Error: Unsupported detected_bbox format: {type(detected_bbox)}")
            return 0.0

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
        return iou