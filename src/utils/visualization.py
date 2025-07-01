"""
Visualization utilities for object tracking and trajectories
"""

import cv2
import numpy as np
import random
import os
from ..config.settings import *


class Visualizer:
    """Handles visualization of objects, trajectories, and tracking information"""

    def __init__(self):
        """Initialize the visualizer"""
        self.object_colors = {}  # Dictionary to store consistent colors for each object ID

    def get_object_color(self, object_id):
        """Get a random but consistent color for an object ID"""
        if object_id not in self.object_colors:
            # Generate random BGR values, but ensure at least one component is bright
            # This helps avoid dark or muddy colors
            while True:
                b = random.randint(0, 255)
                g = random.randint(0, 255)
                r = random.randint(0, 255)

                # Ensure at least one component is bright (above 200)
                # This makes the color more visible
                if max(b, g, r) > 200:
                    break

            self.object_colors[object_id] = (b, g, r)

        return self.object_colors[object_id]

    def draw_trajectories(self, frame, trajectories, frame_width, frame_height):
        """Draw trajectories on the frame"""
        for object_id, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue

            # Get consistent color for this object
            color = self.get_object_color(object_id)

            # Draw trajectory line
            for i in range(1, len(trajectory)):
                start_point = tuple(map(int, trajectory[i-1]))
                end_point = tuple(map(int, trajectory[i]))

                # Only draw if points are within frame
                if (0 <= start_point[0] < frame_width and
                    0 <= start_point[1] < frame_height and
                    0 <= end_point[0] < frame_width and
                    0 <= end_point[1] < frame_height):
                    thickness = TRAJECTORY_THICKNESS
                    cv2.line(frame, start_point, end_point, color, thickness)

            # Draw current position
            if trajectory:
                current_point = tuple(map(int, trajectory[-1]))
                if (0 <= current_point[0] < frame_width and
                    0 <= current_point[1] < frame_height):
                    cv2.circle(frame, current_point, CENTER_POINT_RADIUS, color, -1)

    def draw_complete_trajectories(self, last_frame, full_trajectories, use_kalman=False, kalman_centers=None):
        """Draw complete trajectories on the given frame or reference frame"""
        # Use the provided last frame if available, otherwise use reference frame
        if last_frame is not None:
            trajectory_image = last_frame.copy()
        else:
            print("No frame available for drawing trajectories")
            return None

        # Create a legend image to show object IDs and their colors
        legend_height = min(200, len(full_trajectories) * 25)
        legend_width = 200
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

        # Sort object IDs for consistent legend
        sorted_object_ids = sorted(full_trajectories.keys())

        # Draw all trajectories
        for i, object_id in enumerate(sorted_object_ids):
            trajectory = full_trajectories[object_id]
            if len(trajectory) < 2:
                continue

            # Get consistent color for this object
            color = self.get_object_color(object_id)

            if use_kalman and kalman_centers and object_id in kalman_centers:
                # Draw only Kalman-filtered position
                kalman_x, kalman_y = kalman_centers[object_id]
                cv2.circle(trajectory_image, (int(kalman_x), int(kalman_y)), 4, color, -1)
                cv2.putText(trajectory_image, f"ID: {object_id}",
                           (int(kalman_x) + 10, int(kalman_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # Draw raw trajectory
                x_coords = []
                y_coords = []
                for frame_data in trajectory:
                    if isinstance(frame_data, dict) and 'position' in frame_data:
                        # Handle dictionary format
                        x1, y1 = frame_data['position']['x1'], frame_data['position']['y1']
                        x2, y2 = frame_data['position']['x2'], frame_data['position']['y2']
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                    else:
                        # Handle tuple format
                        center_x, center_y = frame_data
                    x_coords.append(center_x)
                    y_coords.append(center_y)

                # Convert to numpy array for drawing
                pts = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
                if pts.shape[0] > 1:
                    cv2.polylines(trajectory_image, [pts], False, color, 2)

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

                # Draw object ID at the end of trajectory
                cv2.putText(trajectory_image, f"ID: {object_id}",
                           (end_point[0] + 10, end_point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(trajectory_image, f"ID: {object_id}",
                           (end_point[0] + 10, end_point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Add to legend if it fits
            if i < legend_height // 25:
                y_pos = i * 25 + 15
                # Draw color sample
                cv2.rectangle(legend, (10, y_pos-10), (30, y_pos+10), color, -1)
                # Draw ID text
                cv2.putText(legend, f"ID: {object_id}", (40, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add legend to the right side of the trajectory image if there's space
        if trajectory_image.shape[1] > 800:
            x_offset = trajectory_image.shape[1] - legend_width - 10
            y_offset = 10
            if legend_height + y_offset < trajectory_image.shape[0]:
                trajectory_image[y_offset:y_offset+legend_height, x_offset:x_offset+legend_width] = legend

        return trajectory_image

    def draw_objects(self, frame, tracked_objects, frame_width, frame_height,
                    occlusion_handler=None, kalman_manager=None):
        """Draw detections and IDs on frame"""
        for object_id, obj in tracked_objects.items():
            x, y, w, h = obj['bbox']
            center_x, center_y = obj['center']

            # Get consistent color for this object
            color = self.get_object_color(object_id)

            # Check if object is near occlusion area and adjust visualization
            is_near_occlusion = obj.get('near_occlusion', False)
            is_in_occlusion_tracking = obj.get('in_occlusion_tracking', False)
            appeared_from_occlusion = obj.get('appeared_from_occlusion', False)

            if appeared_from_occlusion:
                # Object appeared from roundabout - use green border
                border_color = (0, 255, 0)  # Green border
                border_thickness = 4
            elif is_in_occlusion_tracking:
                # Object is being tracked for occlusion - use red border
                border_color = (0, 0, 255)  # Red border
                border_thickness = 4
            elif is_near_occlusion:
                # Use a different color or add a border for objects near occlusion
                border_color = (0, 255, 255)  # Yellow border
                border_thickness = 3
            else:
                border_color = color
                border_thickness = BOUNDING_BOX_THICKNESS

            # Draw bounding box (allowing it to extend beyond frame boundaries)
            # Calculate visible portion of the box
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_width, x + w)
            y2 = min(frame_height, y + h)

            # Draw the visible portion of the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

            # Draw center point (if it's within the frame)
            if 0 <= center_x < frame_width and 0 <= center_y < frame_height:
                cv2.circle(frame, (center_x, center_y), CENTER_POINT_RADIUS, color, -1)

            # Draw ID and status (if the top of the box is visible)
            if y1 < frame_height:
                id_text = f"ID: {object_id}"
                if appeared_from_occlusion:
                    id_text += " (Exited)"
                elif is_in_occlusion_tracking:
                    id_text += " (Occluded)"
                elif is_near_occlusion:
                    id_text += " (Near)"
                cv2.putText(frame, id_text, (x1, max(10, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw Kalman-filtered positions, velocities, and predicted bounding box
            if kalman_manager and kalman_manager.use_kalman:
                # Get Kalman-filtered center
                kalman_center = kalman_manager.get_kalman_center(object_id)
                if kalman_center:
                    k_x, k_y = kalman_center
                    # Draw Kalman-filtered center point
                    cv2.circle(frame, (k_x, k_y), CENTER_POINT_RADIUS, color, -1)
                    cv2.circle(frame, (k_x, k_y), 6, (255, 255, 255), 1)

                    # Draw line from raw detection to filtered position
                    cv2.line(frame, (center_x, center_y), (k_x, k_y), color, 1, cv2.LINE_AA)

                    # Get and draw predicted position
                    predicted_pos = kalman_manager.get_kalman_prediction(object_id)
                    if predicted_pos:
                        pred_x, pred_y = predicted_pos
                        # Draw predicted bounding box (dashed)
                        pred_x1 = max(0, pred_x - w//2)
                        pred_y1 = max(0, pred_y - h//2)
                        pred_x2 = min(frame_width, pred_x + w//2)
                        pred_y2 = min(frame_height, pred_y + h//2)

                        # Draw dashed rectangle for prediction
                        for i in range(0, w, DASH_LENGTH * 2):
                            if pred_x1 + i < pred_x2:
                                cv2.line(frame,
                                        (pred_x1 + i, pred_y1),
                                        (min(pred_x1 + i + DASH_LENGTH, pred_x2), pred_y1),
                                        color, 1)
                                cv2.line(frame,
                                        (pred_x1 + i, pred_y2),
                                        (min(pred_x1 + i + DASH_LENGTH, pred_x2), pred_y2),
                                        color, 1)

                        for i in range(0, h, DASH_LENGTH * 2):
                            if pred_y1 + i < pred_y2:
                                cv2.line(frame,
                                        (pred_x1, pred_y1 + i),
                                        (pred_x1, min(pred_y1 + i + DASH_LENGTH, pred_y2)),
                                        color, 1)
                                cv2.line(frame,
                                        (pred_x2, pred_y1 + i),
                                        (pred_x2, min(pred_y1 + i + DASH_LENGTH, pred_y2)),
                                        color, 1)

                    # Get and draw velocity
                    velocity_info = kalman_manager.get_kalman_velocity(object_id)
                    if velocity_info:
                        velocity_magnitude, (vx, vy) = velocity_info

                        # Scale velocity vector for visualization
                        scale = VELOCITY_SCALE
                        end_x = int(k_x + vx * scale)
                        end_y = int(k_y + vy * scale)

                        # Draw velocity vector
                        cv2.arrowedLine(frame, (k_x, k_y), (end_x, end_y),
                                       color, 2, tipLength=0.3)

                        # Show velocity magnitude
                        cv2.putText(frame, f"v: {velocity_magnitude:.1f}",
                                   (k_x + 10, k_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_reconstructed_trajectories(self, frame, full_trajectories, frame_width, frame_height):
        """Draw reconstructed trajectory segments in real-time during video playback"""
        for object_id, trajectory in full_trajectories.items():
            if len(trajectory) < 2:
                continue

            # Get consistent color for this object
            color = self.get_object_color(object_id)

            # Find reconstructed segments (frames that were added via ground truth reconstruction)
            reconstructed_frames = []
            for i, frame_data in enumerate(trajectory):
                if isinstance(frame_data, dict) and 'position' in frame_data:
                    # Only include frames that were actually reconstructed from ground truth
                    # if frame_data.get('reconstructed', False):
                    reconstructed_frames.append(frame_data)

            if not reconstructed_frames:
                continue

            # Draw reconstructed trajectory segments
            for i in range(1, len(reconstructed_frames)):
                prev_frame = reconstructed_frames[i-1]
                curr_frame = reconstructed_frames[i]

                # Extract center points
                prev_x1, prev_y1 = prev_frame['position']['x1'], prev_frame['position']['y1']
                prev_x2, prev_y2 = prev_frame['position']['x2'], prev_frame['position']['y2']
                prev_center_x = (prev_x1 + prev_x2) / 2
                prev_center_y = (prev_y1 + prev_y2) / 2

                curr_x1, curr_y1 = curr_frame['position']['x1'], curr_frame['position']['y1']
                curr_x2, curr_y2 = curr_frame['position']['x2'], curr_frame['position']['y2']
                curr_center_x = (curr_x1 + curr_x2) / 2
                curr_center_y = (curr_y1 + curr_y2) / 2

                # Draw reconstructed trajectory line (dashed style)
                start_point = (int(prev_center_x), int(prev_center_y))
                end_point = (int(curr_center_x), int(curr_center_y))

                # Only draw if points are within frame
                if (0 <= start_point[0] < frame_width and
                    0 <= start_point[1] < frame_height and
                    0 <= end_point[0] < frame_width and
                    0 <= end_point[1] < frame_height):

                    # Draw dashed line for reconstructed trajectory
                    dash_length = DASH_LENGTH
                    dx = end_point[0] - start_point[0]
                    dy = end_point[1] - start_point[1]
                    distance = np.sqrt(dx*dx + dy*dy)

                    if distance > 0:
                        # Normalize direction
                        dx = dx / distance
                        dy = dy / distance

                        # Draw dashed line
                        for j in range(0, int(distance), dash_length * 2):
                            seg_start_x = int(start_point[0] + j * dx)
                            seg_start_y = int(start_point[1] + j * dy)
                            seg_end_x = int(start_point[0] + min(j + dash_length, distance) * dx)
                            seg_end_y = int(start_point[1] + min(j + dash_length, distance) * dy)

                            cv2.line(frame, (seg_start_x, seg_start_y), (seg_end_x, seg_end_y),
                                   color, 3, cv2.LINE_AA)
                else:
                    print(f"DEBUG: Points outside frame bounds - start: {start_point}, end: {end_point}, frame: {frame_width}x{frame_height}")

            # Draw reconstructed trajectory points
            for frame_data in reconstructed_frames:
                x1, y1 = frame_data['position']['x1'], frame_data['position']['y1']
                x2, y2 = frame_data['position']['x2'], frame_data['position']['y2']
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if (0 <= center_x < frame_width and 0 <= center_y < frame_height):
                    # Draw reconstructed trajectory points as diamonds
                    cv2.drawMarker(frame, (center_x, center_y), color, cv2.MARKER_DIAMOND, 6, 2)

    def save_individual_trajectories(self, last_frame, full_trajectories, output_dir):
        """Save individual trajectory images for each object"""
        if last_frame is None:
            print("No frame available for drawing individual trajectories")
            return

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Process each object's trajectory
        for object_id, trajectory in full_trajectories.items():
            if len(trajectory) < 2:
                continue

            # Create a copy of the last frame
            trajectory_image = last_frame.copy()

            # Get color for this object
            color = self.get_object_color(object_id)

            # Extract center points from trajectory
            x_coords = []
            y_coords = []
            for frame_data in trajectory:
                if isinstance(frame_data, dict) and 'position' in frame_data:
                    # Handle dictionary format
                    x1, y1 = frame_data['position']['x1'], frame_data['position']['y1']
                    x2, y2 = frame_data['position']['x2'], frame_data['position']['y2']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                else:
                    # Handle tuple format
                    center_x, center_y = frame_data
                x_coords.append(center_x)
                y_coords.append(center_y)

            # Convert to numpy array for drawing
            pts = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)

            # Draw the trajectory
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
            cv2.putText(trajectory_image, f"Object ID: {object_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(trajectory_image, f"Trajectory Points: {len(trajectory)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Save the image
            output_path = os.path.join(output_dir, f"trajectory_object_{object_id}.jpg")
            cv2.imwrite(output_path, trajectory_image)
            print(f"Saved trajectory image for object {object_id} to {output_path}")