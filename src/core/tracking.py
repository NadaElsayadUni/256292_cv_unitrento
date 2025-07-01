"""
Object tracking logic (ID assignment, matching, etc.)
"""

import numpy as np
from collections import defaultdict
from ..config.settings import *


class ObjectTracker:
    """Handles object tracking, ID assignment, and trajectory management"""

    def __init__(self, frame_width, frame_height, original_width=1920, original_height=1080):
        """Initialize the object tracker"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.original_width = original_width
        self.original_height = original_height

        # Tracking parameters
        self.next_object_id = 0
        self.objects = {}
        self.max_disappeared = 150
        self.max_boundary_disappeared = 5
        self.max_distance = 100
        self.frame_count = 0

        # Trajectory management
        self.trajectories = {}  # For real-time visualization (simple tuples)
        self.full_trajectories = {}  # For complete trajectory data (detailed frame data)
        self.max_trajectory_points = 30

    def track_objects(self, detections, frame, occlusion_handler=None, ground_truth_trajectories=None):
        """Track objects across frames and assign consistent IDs"""
        # If no objects are being tracked yet, initialize tracking
        if len(self.objects) == 0:
            for detection in detections:
                x, y, w, h = detection['bbox']
                is_near_boundary = self.is_near_boundary(x, y, w, h)
                self.objects[self.next_object_id] = {
                    'center': detection['center'],
                    'bbox': detection['bbox'],
                    'disappeared': 0,
                    'boundary_disappeared': 0,
                    'near_boundary': is_near_boundary,
                    'last_visible_bbox': detection['bbox'],
                    'last_visible_center': detection['center']
                }
                self.next_object_id += 1
            return self.objects

        # If no detections in current frame, handle all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                if self.objects[object_id]['near_boundary']:
                    self.objects[object_id]['boundary_disappeared'] += 1
                    if self.objects[object_id]['boundary_disappeared'] >= self.max_boundary_disappeared:
                        del self.objects[object_id]
                else:
                    self.objects[object_id]['disappeared'] += 1
                    if self.objects[object_id]['disappeared'] >= self.max_disappeared:
                        del self.objects[object_id]
            return self.objects

        # Check for objects near occlusion area
        if occlusion_handler:
            # print(f"Handling occlusion tracking for {len(self.objects)} objects")
            self._handle_occlusion_tracking(occlusion_handler)

        # Calculate distances between existing objects and new detections
        # Filter out occluded objects from tracking
        occluded_objects = occlusion_handler.occluded_objects if occlusion_handler else {}
        object_ids = [obj_id for obj_id in self.objects.keys() if obj_id not in occluded_objects]
        object_centers = [self.objects[obj_id]['center'] for obj_id in object_ids]
        detection_centers = [det['center'] for det in detections]

        # If we have existing objects, try to match them with new detections
        if len(object_centers) > 0:
            # Calculate distances between all pairs of existing objects and new detections
            distances = []
            for i, obj_center in enumerate(object_centers):
                for j, det_center in enumerate(detection_centers):
                    distance = np.sqrt((obj_center[0] - det_center[0])**2 +
                                     (obj_center[1] - det_center[1])**2)
                    distances.append((i, j, distance))

            # Sort distances
            distances.sort(key=lambda x: x[2])

            # Match objects with detections
            matched_objects = set()
            matched_detections = set()
            for i, j, distance in distances:
                if distance > self.max_distance:
                    continue
                if i not in matched_objects and j not in matched_detections:
                    x, y, w, h = detections[j]['bbox']
                    is_near_boundary = self.is_near_boundary(x, y, w, h)

                    # Update object state
                    self.objects[object_ids[i]].update({
                        'center': detection_centers[j],
                        'bbox': detections[j]['bbox'],
                        'near_boundary': is_near_boundary,
                        'last_visible_bbox': detections[j]['bbox'],
                        'last_visible_center': detection_centers[j]
                    })

                    # Reset appropriate counters
                    if is_near_boundary:
                        self.objects[object_ids[i]]['boundary_disappeared'] = 0
                    else:
                        self.objects[object_ids[i]]['disappeared'] = 0
                        self.objects[object_ids[i]]['boundary_disappeared'] = 0

                    matched_objects.add(i)
                    matched_detections.add(j)

            # Handle unmatched objects
            for i in range(len(object_centers)):
                if i not in matched_objects:
                    obj_id = object_ids[i]
                    # Skip occluded objects
                    if obj_id in occluded_objects:
                        continue

                    if self.objects[obj_id]['near_boundary']:
                        self.objects[obj_id]['boundary_disappeared'] += 1
                        if self.objects[obj_id]['boundary_disappeared'] > self.max_boundary_disappeared:
                            del self.objects[obj_id]
                    else:
                        self.objects[obj_id]['disappeared'] += 1
                        if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                            del self.objects[obj_id]

            # Handle new detections
            for j in range(len(detection_centers)):
                if j not in matched_detections:
                    x, y, w, h = detections[j]['bbox']
                    center_x, center_y = detection_centers[j]
                    is_near_boundary = self.is_near_boundary(x, y, w, h)

                    # Check if new detection appears near occlusion area
                    if occlusion_handler and occlusion_handler.is_point_near_occlusion(center_x, center_y, proximity_distance=OCCLUSION_PROXIMITY_DISTANCE):
                        # print(f"New detection near occlusion area at frame {self.frame_count} (position: {center_x:.1f}, {center_y:.1f})")
                        # Create object with occlusion tracking initialized
                        new_obj = {
                            'center': detection_centers[j],
                            'bbox': detections[j]['bbox'],
                            'disappeared': 0,
                            'boundary_disappeared': 0,
                            'near_boundary': is_near_boundary,
                            'last_visible_bbox': detections[j]['bbox'],
                            'last_visible_center': detection_centers[j],
                            'near_occlusion': True,
                            'first_near_occlusion_frame': self.frame_count,
                            'stationary_start_position': (center_x, center_y),
                            'stationary_frames': 0,
                            'appeared_from_occlusion': True  # Mark as appeared from roundabout
                        }

                        # Try to match with occluded objects
                        if ground_truth_trajectories:
                            isAssignedOccludedObject = occlusion_handler.track_occluded_objects(
                                new_obj, ground_truth_trajectories, self.full_trajectories,
                                self.frame_width, self.frame_height, self.original_width, self.original_height
                            )
                        else:
                            isAssignedOccludedObject = (None, None, None, 0.0)

                        if isAssignedOccludedObject[0] == None:  # No match found
                            # No match found - create new object
                            self.objects[self.next_object_id] = new_obj
                            self.next_object_id += 1
                        else:
                            # Match found - unpack the results
                            occluded_object_id, matched_gt_id, matched_gt_trajectory, gt_match_score = isAssignedOccludedObject

                            # Update existing object and remove from occluded tracking
                            if occluded_object_id in occlusion_handler.occluded_objects:
                                self.objects[occluded_object_id].update({
                                    'center': new_obj['center'],
                                    'bbox': new_obj['bbox'],
                                    'last_visible_bbox': new_obj['bbox'],
                                    'last_visible_center': new_obj['center'],
                                    'disappeared': 0,
                                    'boundary_disappeared': 0,
                                    'appeared_from_occlusion': True  # Mark that it reappeared from occlusion
                                })
                                # Remove from occluded objects since it's now visible again
                                del occlusion_handler.occluded_objects[occluded_object_id]
                                print(f"Removed object {occluded_object_id} from occluded tracking")

                                # Reconstruct missing trajectory from ground truth
                                if matched_gt_id is not None and matched_gt_trajectory is not None:
                                    print(f"\n=== TRAJECTORY RECONSTRUCTION ===")
                                    print(f"✅ SUCCESS: Reconstructing trajectory for object {occluded_object_id}")
                                    print(f"   Matched to ground truth {matched_gt_id} with score {gt_match_score:.3f}")
                                    print(f"   Ground truth trajectory has {len(matched_gt_trajectory)} frames")

                                    # Reconstruct missing trajectory frames
                                    frames_added = self.reconstruct_missing_trajectory(
                                        occluded_object_id,
                                        matched_gt_trajectory,
                                        new_obj['first_near_occlusion_frame']
                                    )

                                    if frames_added > 0:
                                        print(f"✅ Successfully added {frames_added} missing trajectory frames")
                                    else:
                                        print("⚠️ No missing frames to reconstruct")

                        print(f"Assigned new object {isAssignedOccludedObject[0] if isAssignedOccludedObject[0] is not None else 'new'} to occlusion tracking (appeared from roundabout)")

                    else:
                        # Regular new detection (not near occlusion)
                        self.objects[self.next_object_id] = {
                            'center': detection_centers[j],
                            'bbox': detections[j]['bbox'],
                            'disappeared': 0,
                            'boundary_disappeared': 0,
                            'near_boundary': is_near_boundary,
                            'last_visible_bbox': detections[j]['bbox'],
                            'last_visible_center': detection_centers[j]
                        }
                        self.next_object_id += 1

        return self.objects

    def _handle_occlusion_tracking(self, occlusion_handler):
        """Handle occlusion tracking for existing objects"""
        for object_id, obj in self.objects.items():
            if object_id not in occlusion_handler.occluded_objects:  # Only check non-occluded objects
                appeared_from_occlusion = obj.get('appeared_from_occlusion', False)
                if appeared_from_occlusion:
                    obj['appeared_from_occlusion_counter'] = obj.get('appeared_from_occlusion_counter', 0) + 1
                    if obj['appeared_from_occlusion_counter'] >= OCCLUSION_APPEARED_COUNTER:
                        del obj['appeared_from_occlusion']
                        del obj['appeared_from_occlusion_counter']
                        print(f"Object {object_id} removing from occluded tracking")
                    continue
                center_x, center_y = obj['center']
                if occlusion_handler.is_point_near_occlusion(center_x, center_y, proximity_distance=OCCLUSION_PROXIMITY_DISTANCE):
                    print(f"Object {object_id} is near occlusion area at frame {self.frame_count} (position: {center_x:.1f}, {center_y:.1f})")

                    # Check if object is staying at the same position
                    if 'near_occlusion' not in obj:
                        # First time near occlusion - initialize tracking
                        obj['near_occlusion'] = True
                        obj['first_near_occlusion_frame'] = self.frame_count
                        obj['stationary_start_position'] = (center_x, center_y)
                        obj['stationary_frames'] = 0
                        print(f"Object {object_id} started tracking near occlusion at position ({center_x:.1f}, {center_y:.1f}) frame {self.frame_count}")
                    else:
                        # Object is already near occlusion - check if it's stationary
                        stationary_start_x, stationary_start_y = obj['stationary_start_position']
                        distance_moved = np.sqrt((center_x - stationary_start_x)**2 + (center_y - stationary_start_y)**2)

                        # If object hasn't moved much (less than threshold), increment stationary counter
                        if distance_moved < OCCLUSION_STATIONARY_THRESHOLD:
                            obj['stationary_frames'] += 1
                            print(f"Object {object_id} stationary for {obj['stationary_frames']} frames near occlusion")

                            # If object has been stationary for more than threshold frames, add to occluded_objects
                            if obj['stationary_frames'] >= OCCLUSION_STATIONARY_FRAMES:
                                print(f"Object {object_id} has been stationary near occlusion for {obj['stationary_frames']} frames - adding to occluded tracking frame {self.frame_count}")
                                obj['in_occlusion_tracking'] = True
                                obj['last_near_occlusion_frame'] = self.frame_count
                                obj['last_near_occlusion_bbox'] = obj['bbox']
                                occlusion_handler.occluded_objects[object_id] = obj.copy()

                        else:
                            # Object moved significantly - reset stationary tracking
                            obj['stationary_start_position'] = (center_x, center_y)
                            obj['stationary_frames'] = 0
                            print(f"Object {object_id} moved from stationary position, resetting tracking")
                else:
                    # Object is no longer near occlusion
                    if 'near_occlusion' in obj:
                        del obj['near_occlusion']
                        if 'first_near_occlusion_frame' in obj:
                            del obj['first_near_occlusion_frame']
                        if 'stationary_start_position' in obj:
                            del obj['stationary_start_position']
                        if 'stationary_frames' in obj:
                            del obj['stationary_frames']
                        if 'in_occlusion_tracking' in obj:
                            del obj['in_occlusion_tracking']
                        if object_id in occlusion_handler.occluded_objects:
                            del occlusion_handler.occluded_objects[object_id]
                        print(f"Object {object_id} is no longer near occlusion area at frame {self.frame_count}")

    def is_near_boundary(self, x, y, w, h):
        """Check if object is near the frame boundary"""
        margin = 20  # pixels from boundary
        return (x <= margin or
                y <= margin or
                x + w >= self.frame_width - margin or
                y + h >= self.frame_height - margin)

    def update_trajectories(self, objects):
        """Update trajectories for all tracked objects"""
        for object_id, obj in objects.items():
            # Initialize trajectory if it doesn't exist
            if object_id not in self.trajectories:
                self.trajectories[object_id] = []
            if object_id not in self.full_trajectories:
                self.full_trajectories[object_id] = []

            # Add current position to trajectory
            self.trajectories[object_id].append(obj['center'])
            self.full_trajectories[object_id].append(obj['center'])

        # Remove trajectories for objects that no longer exist
        for object_id in list(self.trajectories.keys()):
            if object_id not in objects:
                del self.trajectories[object_id]
                if object_id in self.full_trajectories:
                    del self.full_trajectories[object_id]

    def update_trajectories_with_kalman(self, objects, kalman_manager, occlusion_handler=None):
        """Update trajectories with Kalman-filtered positions if available"""
        for object_id, obj in objects.items():
            # Skip occluded objects - don't add trajectory frames during occlusion
            if occlusion_handler and object_id in occlusion_handler.occluded_objects:
                continue

            # Initialize trajectory if it doesn't exist
            if object_id not in self.full_trajectories:
                self.full_trajectories[object_id] = []

            # Get position - either Kalman-filtered or raw detection
            kalman_center = kalman_manager.get_kalman_center(object_id)
            if kalman_manager.use_kalman and kalman_center:
                # Use Kalman-filtered position
                position = kalman_center
            else:
                # Use raw detection
                position = obj['center']

            # Create frame data matching readFiles.py structure
            x, y = position
            x1, y1, w, h = obj['bbox']

            frame_data = {
                'frame_number': self.frame_count,
                'position': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x1 + w),
                    'y2': float(y1 + h)
                },
                'metadata': [0, 0, 0],  # Default metadata
                'class': 'Unknown'  # Default class
            }

            # Add to full trajectories
            self.full_trajectories[object_id].append(frame_data)

            # Add position to real-time trajectory
            if object_id not in self.trajectories:
                self.trajectories[object_id] = []
            self.trajectories[object_id].append((x, y))  # Store as tuple of (x,y)

            # Limit trajectory points for real-time visualization
            if len(self.trajectories[object_id]) > self.max_trajectory_points:
                self.trajectories[object_id] = self.trajectories[object_id][-self.max_trajectory_points:]

    def reconstruct_missing_trajectory(self, object_id, matched_gt_trajectory, new_detection_frame):
        """
        Reconstruct missing trajectory frames from ground truth data.

        Args:
            object_id: ID of the object to reconstruct trajectory for
            matched_gt_trajectory: The matched ground truth trajectory
            new_detection_frame: Frame number where the object reappeared

        Returns:
            Number of frames added to the trajectory
        """
        print(f"\n=== RECONSTRUCTING MISSING TRAJECTORY FOR OBJECT {object_id} ===")

        # Get the object's current trajectory
        if object_id not in self.full_trajectories:
            self.full_trajectories[object_id] = []
            print(f"No trajectory found for object {object_id}, creating new one")
            return 0

        current_trajectory = self.full_trajectories[object_id]
        if not current_trajectory:
            print(f"Empty trajectory for object {object_id}")
            return 0

        # Find the last frame in the current trajectory
        last_frame_in_trajectory = max(frame['frame_number'] for frame in current_trajectory)
        print(f"Last frame in current trajectory: {last_frame_in_trajectory}")
        print(f"New detection frame: {new_detection_frame}")

        # Find frames in ground truth that are between the last trajectory frame and new detection
        missing_frames = []
        for frame_data in matched_gt_trajectory:
            frame_num = frame_data['frame_number']
            if last_frame_in_trajectory < frame_num < new_detection_frame:
                missing_frames.append(frame_data)

        if not missing_frames:
            print(f"No missing frames found between {last_frame_in_trajectory} and {new_detection_frame}")
            return 0

        print(f"Found {len(missing_frames)} missing frames to reconstruct")

        # Add the missing frames to the trajectory
        frames_added = 0
        for frame_data in missing_frames:
            # Convert ground truth coordinates to video resolution using correct original dimensions
            x_scale = self.frame_width / self.original_width
            y_scale = self.frame_height / self.original_height

            # Scale the ground truth coordinates
            x1 = frame_data['position']['x1'] * x_scale
            y1 = frame_data['position']['y1'] * y_scale
            x2 = frame_data['position']['x2'] * x_scale
            y2 = frame_data['position']['y2'] * y_scale

            # Create trajectory frame data
            trajectory_frame = {
                'frame_number': frame_data['frame_number'],
                'position': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                },
                'metadata': frame_data.get('metadata', [0, 0, 0]),
                'class': frame_data.get('class', 'Unknown'),
                'reconstructed': True  # Flag to identify reconstructed frames
            }

            # Add to trajectory
            self.full_trajectories[object_id].append(trajectory_frame)
            frames_added += 1

            # print(f"  Added frame {frame_data['frame_number']}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

        # print(f"✅ Successfully reconstructed {frames_added} missing trajectory frames for object {object_id}")
        return frames_added

    def set_frame_count(self, frame_count):
        """Set the current frame count"""
        self.frame_count = frame_count