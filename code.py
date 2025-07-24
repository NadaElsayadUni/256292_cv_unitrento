import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from readFiles import read_annotations_file
from collections import defaultdict  # Add this import

def compute_occlusion_mask(with_occlusion_path, without_occlusion_path, save_path="occlusion_mask.png"):
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

class HumanMotionAnalyzer:
    def __init__(self, video_path, reference_path, annotations_path):
        self.video_path = video_path
        self.reference_path = reference_path
        self.annotations_path = annotations_path
        self.cap = None
        self.reference_frame = None
        self.trajectories = {}  # Dictionary to store trajectory points
        self.full_trajectories = defaultdict(list)  # Changed to defaultdict(list)
        self.frame_count = 0  # Add frame counter

        # Read ground truth to determine original resolution
        self.ground_truth_trajectories, _, _ = read_annotations_file(self.annotations_path)
        if self.ground_truth_trajectories:
            # Get max coordinates from ground truth to determine resolution
            max_x = max_y = 0
            for trajectory in self.ground_truth_trajectories.values():
                for frame in trajectory:
                    max_x = max(max_x, frame['position']['x2'])
                    max_y = max(max_y, frame['position']['y2'])

            # Use exact ground truth resolution without padding
            self.original_width = int(max_x)
            self.original_height = int(max_y)
            print(f"Determined original resolution from ground truth: {self.original_width}x{self.original_height}")
            print(f"Loaded {len(self.ground_truth_trajectories)} complete ground truth trajectories for reconstruction")
        else:
            # Default values if ground truth can't be read
            self.original_width = 1920
            self.original_height = 1080
            print("Warning: Could not read ground truth, using default resolution")


        # Initialize background subtractor with stricter parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=20,  # Increased threshold to be more strict
            detectShadows=False
        )
        # Parameters for object detection
        self.min_area = 600  # Increased minimum area to filter out more noise
        self.max_area = 10000  # Maximum area for detected objects
        # Tracking parameters
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store object information
        self.max_disappeared = 150  # Maximum number of frames an object can be missing
        self.max_boundary_disappeared = 5  # Faster disappearance for objects near boundaries
        self.max_distance = 100  # Maximum distance to consider objects as the same
        self.frame_width = None
        self.frame_height = None
        # Trajectory parameters
        self.max_trajectory_points = 30  # Maximum number of points to keep in trajectory for display
        # Color management
        self.object_colors = {}  # Dictionary to store consistent colors for each object ID

        # Kalman filter for each object
        self.kalman_filters = {}  # Dictionary to store Kalman filter for each object ID
        self.kalman_initialized = {}  # Track which filters have been initialized
        self.kalman_centers = {}  # Track Kalman-filtered positions
        self.use_kalman = True  # Flag to enable/disable Kalman filtering
         # Add occlusion tracking
        self.occlusion_mask = None
        self.occluded_objects = {}  # Dictionary to store objects in occlusion
        self.prev_positions = {}    # Store previous positions for velocity calculation

    def track_objects(self, detections, frame):
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
        for object_id, obj in self.objects.items():
            if object_id not in self.occluded_objects:  # Only check non-occluded objects
                appeared_from_occlusion = obj.get('appeared_from_occlusion', False)
                if appeared_from_occlusion:
                    # print(f"Object {object_id} is new detected from occluded objects at frame {self.frame_count} stop occluded and start tracking")
                    obj['appeared_from_occlusion_counter'] = obj.get('appeared_from_occlusion_counter', 0) + 1
                    if obj['appeared_from_occlusion_counter'] >= 10:
                        del obj['appeared_from_occlusion']
                        del obj['appeared_from_occlusion_counter']
                        print(f"Object {object_id} removing from occluded tracking")
                    continue
                center_x, center_y = obj['center']
                if self.is_point_near_occlusion(center_x, center_y, proximity_distance=20):
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

                        # If object hasn't moved much (less than 10 pixels), increment stationary counter
                        if distance_moved < 10:
                            obj['stationary_frames'] += 1
                            print(f"Object {object_id} stationary for {obj['stationary_frames']} frames near occlusion")

                            # If object has been stationary for more than 5 frames, add to occluded_objects
                            if obj['stationary_frames'] >= 5:
                                print(f"Object {object_id} has been stationary near occlusion for {obj['stationary_frames']} frames - adding to occluded tracking frame {self.frame_count}")
                                obj['in_occlusion_tracking'] = True
                                obj['last_near_occlusion_frame'] = self.frame_count
                                obj['last_near_occlusion_bbox'] = obj['bbox']
                                self.occluded_objects[object_id] = obj.copy()

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
                        if object_id in self.occluded_objects:
                            del self.occluded_objects[object_id]
                        print(f"Object {object_id} is no longer near occlusion area at frame {self.frame_count}")

        # Calculate distances between existing objects and new detections
        # Filter out occluded objects from tracking
        object_ids = [obj_id for obj_id in self.objects.keys() if obj_id not in self.occluded_objects]
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
                    if obj_id in self.occluded_objects:
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
                    if self.is_point_near_occlusion(center_x, center_y, proximity_distance=20):
                        print(f"New detection near occlusion area at frame {self.frame_count} (position: {center_x:.1f}, {center_y:.1f})")
                        # go to occluded_objects and get the object_id
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
                        isAssignedOccludedObject = self.track_occluded_objects(new_obj)
                        if isAssignedOccludedObject[0] == None:  # No match found
                            # No match found - create new object
                            self.objects[self.next_object_id] = new_obj
                            self.next_object_id += 1
                        else:
                            # Match found - unpack the results
                            occluded_object_id, matched_gt_id, matched_gt_trajectory, gt_match_score = isAssignedOccludedObject

                            # Update existing object and remove from occluded tracking
                            if occluded_object_id in self.occluded_objects:
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
                                del self.occluded_objects[occluded_object_id]
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

    def initialize_kalman_filter(self):
        """Initialize a new Kalman filter for tracking"""
        kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)

        # Increased process noise for better handling of occlusions
        kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32) * 0.1  # Increased from 0.03

        # Decreased measurement noise to trust measurements more when available
        kalman.measurementNoiseCov = np.array([[1, 0],
                                             [0, 1]], np.float32) * 0.005  # Decreased from 0.01

        # Initialize error covariance
        kalman.errorCovPost = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], np.float32) * 0.1

        return kalman

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

    def load_reference_frame(self):
        """Load the reference frame (background)"""
        self.reference_frame = cv2.imread(self.reference_path)
        if self.reference_frame is None:
            raise ValueError(f"Could not load reference frame from {self.reference_path}")
        return self.reference_frame

    def load_video(self):
        """Load the video file"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file {self.video_path}")
        # Get video dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nVideo resolution: {self.frame_width}x{self.frame_height}")
        return self.cap

    def detect_moving_objects(self, frame):
        """Detect moving objects using background subtraction"""
        # Add at start of method

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Apply initial morphological operations to remove noise
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Apply additional noise reduction
        fg_mask = cv2.GaussianBlur(fg_mask, (5,5), 0)
        _, fg_mask = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)

        # Make objects more dense using dilation
        kernel_dilate = np.ones((7,7), np.uint8)
        fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=1)

        # Additional morphological operations to clean up objects
        kernel_open = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

        # Final noise removal with larger kernel
        kernel_clean = np.ones((7,7), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_clean)
        _,fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and process detected objects
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # print(f"Raw detection bbox: x={x}, y={y}, w={w}, h={h}")
                # Calculate center point
                center_x = x + w//2
                center_y = y + h//2

                detections.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area
                })

        return detections, fg_mask

    def is_near_boundary(self, x, y, w, h):
        """Check if object is near the frame boundary"""
        margin = 20  # pixels from boundary
        return (x <= margin or
                y <= margin or
                x + w >= self.frame_width - margin or
                y + h >= self.frame_height - margin)

    def update_kalman_filters(self, tracked_objects):
        """Update Kalman filters for all tracked objects after regular tracking"""
        if not self.use_kalman:
            return

        # Process each tracked object
        for object_id, obj in tracked_objects.items():
            center_x, center_y = obj['center']
            # Create and initialize filter if it doesn't exist
            if object_id not in self.kalman_filters:
                self.kalman_filters[object_id] = self.initialize_kalman_filter()
                # Initialize state with first measurement
                self.kalman_filters[object_id].statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
                self.kalman_initialized[object_id] = True
                self.kalman_centers[object_id] = (center_x, center_y)
                continue

            # For existing filters, predict and then correct with new measurement
            kalman = self.kalman_filters[object_id]

            # First predict
            kalman.predict()

            # Then correct with measurement
            measurement = np.array([[center_x], [center_y]], np.float32)
            kalman.correct(measurement)

            # Store the filtered position
            filtered_x = int(kalman.statePost[0][0])
            filtered_y = int(kalman.statePost[1][0])
            self.kalman_centers[object_id] = (filtered_x, filtered_y)

        # Clean up filters for objects that no longer exist
        for object_id in list(self.kalman_filters.keys()):
            if object_id not in tracked_objects:
                del self.kalman_filters[object_id]
                if object_id in self.kalman_initialized:
                    del self.kalman_initialized[object_id]
                if object_id in self.kalman_centers:
                    del self.kalman_centers[object_id]

    def get_kalman_center(self, object_id):
        """Get Kalman-filtered center position for an object, if available"""
        if not self.use_kalman or object_id not in self.kalman_centers:
            return None
        return self.kalman_centers[object_id]

    def get_kalman_velocity(self, object_id):
        """Get velocity from Kalman filter for an object, if available"""
        if not self.use_kalman or object_id not in self.kalman_filters:
            return None

        kalman = self.kalman_filters[object_id]
        vx = kalman.statePost[2][0]
        vy = kalman.statePost[3][0]
        velocity_magnitude = np.sqrt(vx**2 + vy**2)

        return velocity_magnitude, (vx, vy)

    def get_kalman_prediction(self, object_id):
        """Get predicted position from Kalman filter for an object"""
        if not self.use_kalman or object_id not in self.kalman_filters:
            return None

        kalman = self.kalman_filters[object_id]
        # Get the predicted state
        prediction = kalman.predict().copy()  # Make a copy to not affect the actual filter
        # Reset the filter state since we don't want this prediction to affect the actual tracking
        kalman.statePost = kalman.statePre.copy()

        # Extract predicted x, y coordinates
        pred_x = int(prediction[0][0])
        pred_y = int(prediction[1][0])

        return (pred_x, pred_y)

    def update_trajectories_with_kalman(self, objects):
        """Update trajectories with Kalman-filtered positions if available"""
        for object_id, obj in objects.items():
            # Skip occluded objects - don't add trajectory frames during occlusion
            if object_id in self.occluded_objects:
                continue

            # Get position - either Kalman-filtered or raw detection
            kalman_center = self.get_kalman_center(object_id)
            if self.use_kalman and kalman_center:
                # Use Kalman-filtered position
                # print(f"Kalman-filtered position for object {object_id} x: {kalman_center[0]} y: { kalman_center[1] }")
                position = kalman_center
            else:
                # Use raw detection
                # print(f"Raw detection for object {object_id} x: {obj['center'][0]} y: { obj['center'][1] }")
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

    def update_trajectories(self, objects):
        """Update trajectories for all tracked objects"""
        for object_id, obj in objects.items():
            # Initialize trajectory if it doesn't exist
            if object_id not in self.trajectories:
                self.trajectories[object_id] = []
                self.full_trajectories[object_id] = []

            # Add current position to trajectory
            self.trajectories[object_id].append(obj['center'])
            self.full_trajectories[object_id].append(obj['center'])

            # limiting trajectory points
            # if len(self.trajectories[object_id]) > self.max_trajectory_points:
            #     self.trajectories[object_id] = self.trajectories[object_id][-self.max_trajectory_points:]

        # Remove trajectories for objects that no longer exist
        for object_id in list(self.trajectories.keys()):
            if object_id not in objects:
                del self.trajectories[object_id]

    def draw_trajectories(self, frame):
        """Draw trajectories on the frame"""
        for object_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue

            # Get consistent color for this object
            color = self.get_object_color(object_id)

            # Draw trajectory line
            for i in range(1, len(trajectory)):
                start_point = tuple(map(int, trajectory[i-1]))
                end_point = tuple(map(int, trajectory[i]))

                # Only draw if points are within frame
                if (0 <= start_point[0] < self.frame_width and
                    0 <= start_point[1] < self.frame_height and
                    0 <= end_point[0] < self.frame_width and
                    0 <= end_point[1] < self.frame_height):
                    thickness = 2
                    cv2.line(frame, start_point, end_point, color, thickness)

            # Draw current position
            if trajectory:
                current_point = tuple(map(int, trajectory[-1]))
                if (0 <= current_point[0] < self.frame_width and
                    0 <= current_point[1] < self.frame_height):
                    cv2.circle(frame, current_point, 4, color, -1)

    def draw_complete_trajectories(self, last_frame=None, use_kalman=False):
        """Draw complete trajectories on the given frame or reference frame"""
        # Use the provided last frame if available, otherwise use reference frame
        if last_frame is not None:
            trajectory_image = last_frame.copy()
        elif self.reference_frame is not None:
            trajectory_image = self.reference_frame.copy()
        else:
            print("No frame available for drawing trajectories")
            return None

        # Create a legend image to show object IDs and their colors
        legend_height = min(200, len(self.full_trajectories) * 25)
        legend_width = 200
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

        # Sort object IDs for consistent legend
        sorted_object_ids = sorted(self.full_trajectories.keys())

        # Draw all trajectories
        for i, object_id in enumerate(sorted_object_ids):
            trajectory = self.full_trajectories[object_id]
            if len(trajectory) < 2:
                continue

            # Get consistent color for this object
            color = self.get_object_color(object_id)

            if use_kalman and self.use_kalman and object_id in self.kalman_centers:
                # Draw only Kalman-filtered position
                kalman_x, kalman_y = self.kalman_centers[object_id]
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

    def _calculate_trajectory_mse(self, gt_trajectory, detected_trajectory):
        """Calculate Mean Squared Error between two trajectories using bounding box coordinates"""
        # Create dictionaries for easy lookup by frame number
        gt_dict = {frame['frame_number']: frame for frame in gt_trajectory}
        detected_dict = {frame['frame_number']: frame for frame in detected_trajectory}

        # Find common frames
        common_frames = set(gt_dict.keys()) & set(detected_dict.keys())
        if not common_frames:
            return float('inf')

        # Need at least 5 common frames to consider it a valid match
        if len(common_frames) < 5:
            return float('inf')

        # Calculate scaling factors to convert ground truth to video resolution
        x_scale = self.frame_width / self.original_width
        y_scale = self.frame_height / self.original_height

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

            # print(f"\nFrame {frame_num} MSE calculation:")
            # print(f"Ground truth box: ({gt_x1:.1f}, {gt_y1:.1f}, {gt_x2:.1f}, {gt_y2:.1f})")
            # print(f"Detected box: ({det_x1:.1f}, {det_y1:.1f}, {det_x2:.1f}, {det_y2:.1f})")
            # print(f"Ground truth size: {gt_width:.1f}x{gt_height:.1f}")
            # print(f"Detected size: {det_width:.1f}x{det_height:.1f}")
            # print(f"Frame error: {error:.3f}")

        # Return normalized MSE (average error per frame)
        avg_error = total_error / len(common_frames)
        print(f"\nAverage MSE across {len(common_frames)} frames: {avg_error:.3f}")
        return avg_error

    def _calculate_trajectory_iou(self, gt_trajectory, detected_trajectory):
        """Calculate Intersection over Union (IoU) between two trajectories"""
        # Create dictionaries for easy lookup by frame number
        gt_dict = {frame['frame_number']: frame for frame in gt_trajectory}
        detected_dict = {frame['frame_number']: frame for frame in detected_trajectory}

        # Find common frames
        common_frames = set(gt_dict.keys()) & set(detected_dict.keys())
        if not common_frames:
            return 0.0

        # Need at least 5 common frames to consider it a valid match
        if len(common_frames) < 5:
            return 0.0

        # Calculate scaling factors to convert ground truth to video resolution
        x_scale = self.frame_width / self.original_width
        y_scale = self.frame_height / self.original_height

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

            # print(f"\nFrame {frame_num} IoU: {iou:.3f}")
            # print(f"Ground truth box: ({gt_x1:.1f}, {gt_y1:.1f}, {gt_x2:.1f}, {gt_y2:.1f})")
            # print(f"Detected box: ({det_x1:.1f}, {det_y1:.1f}, {det_x2:.1f}, {det_y2:.1f})")
            # print(f"Intersection area: {inter_area:.1f}")
            # print(f"Union area: {union_area:.1f}")

        # Return average IoU
        avg_iou = total_iou / len(common_frames)
        print(f"\nAverage IoU across {len(common_frames)} frames: {avg_iou:.3f}")
        return avg_iou

    def compute_trajectory_accuracy(self):
        """Compute accuracy metrics between detected and ground truth trajectories"""
        # Read ground truth trajectories from annotations file
        ground_truth_trajectories, _, _ = read_annotations_file(self.annotations_path)

        if not ground_truth_trajectories:
            print("Error: Could not read ground truth trajectories")
            return None

        print(f"\nDebug: Found {len(ground_truth_trajectories)} ground truth trajectories")
        print(f"Debug: Found {len(self.full_trajectories)} detected trajectories")

        # Dictionary to store accuracy metrics
        accuracy_metrics = {
            'mse': {},  # Mean Squared Error for each object
            'iou': {},  # Intersection over Union for each object
            'matched_objects': 0,  # Number of objects that were successfully tracked
            'total_objects': len(ground_truth_trajectories),
            'average_mse': 0.0,
            'average_iou': 0.0
        }

        # For each ground truth trajectory, find the best matching detected trajectory
        for gt_id, gt_trajectory in ground_truth_trajectories.items():
            best_iou = 0.0  # Start with 0 IoU
            best_match_id = None

            # First find the best match using IoU
            for detected_id, detected_trajectory in self.full_trajectories.items():
                # Calculate IoU for matching
                iou = self._calculate_trajectory_iou(gt_trajectory, detected_trajectory)
                print(f"\nComparing ground truth {gt_id} with detected {detected_id}:")
                print(f"IoU: {iou:.3f}")

                # Update best match if this IoU is better (no threshold)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = detected_id

            # If we found a good match, calculate MSE for that match
            if best_match_id is not None:
                print(f"\nFound best match for ground truth {gt_id}: detected {best_match_id}")
                print(f"Best IoU: {best_iou:.3f}")

                # Calculate MSE for the best match
                mse = self._calculate_trajectory_mse(gt_trajectory, self.full_trajectories[best_match_id])
                print(f"MSE for best match: {mse:.3f}")

                accuracy_metrics['mse'][gt_id] = {
                    'detected_id': best_match_id,
                    'mse': mse
                }
                accuracy_metrics['iou'][gt_id] = {
                    'detected_id': best_match_id,
                    'iou': best_iou
                }
                if mse <= 15 and best_iou >= 0.1:
                    accuracy_metrics['matched_objects'] += 1
            else:
                print(f"\nNo good match found for ground truth object {gt_id}")

        # Calculate average MSE and IoU
        if accuracy_metrics['matched_objects'] > 0:
            accuracy_metrics['average_mse'] = sum(m['mse'] for m in accuracy_metrics['mse'].values() if m['mse'] <= 15) / accuracy_metrics['matched_objects']
            accuracy_metrics['average_iou'] = sum(i['iou'] for i in accuracy_metrics['iou'].values() if i['iou'] >= 0.1) / accuracy_metrics['matched_objects']

        print("\n=== Final Accuracy Metrics ===")
        print(f"Total ground truth objects: {accuracy_metrics['total_objects']}")
        print(f"Successfully matched objects: {accuracy_metrics['matched_objects']}")
        print(f"Average IoU: {accuracy_metrics['average_iou']:.3f}")
        print(f"Average MSE: {accuracy_metrics['average_mse']:.3f}")

        print("\nDetailed metrics per object:")
        for gt_id, metrics in accuracy_metrics['mse'].items():
            detected_id = metrics['detected_id']
            mse = metrics['mse']
            iou = accuracy_metrics['iou'][gt_id]['iou']
            print(f"Ground truth ID {gt_id} -> Detected ID {detected_id}: MSE = {mse:.3f}, IoU = {iou:.3f}")

        return accuracy_metrics

    def save_individual_trajectories(self, last_frame):
        """Save individual trajectory images for each object"""
        if last_frame is None:
            print("No frame available for drawing individual trajectories")
            return

        # Create output directory if it doesn't exist
        output_dir = "detected_trajectories"
        os.makedirs(output_dir, exist_ok=True)

        # Process each object's trajectory
        for object_id, trajectory in self.full_trajectories.items():
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
                x1, y1 = frame_data['position']['x1'], frame_data['position']['y1']
                x2, y2 = frame_data['position']['x2'], frame_data['position']['y2']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
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
            # print(f"Saved trajectory image for object {object_id} to {output_path}")

    def save_trajectories_to_file(self, output_file="trajectories.txt"):
        """Save full trajectories data to a file in the specified format:
        Track ID xmin ymin xmax ymax frame
        """
        with open(output_file, 'w') as f:
            # Sort trajectories by track ID for consistent output
            for track_id in sorted(self.full_trajectories.keys()):
                trajectory = self.full_trajectories[track_id]
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
        cv2.imwrite("aligned_occlusion_mask.png", self.occlusion_mask)
        print("Aligned occlusion mask saved as 'aligned_occlusion_mask.png'")

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
        # print(f"Occlusion mask shape: {self.occlusion_mask.shape},x: {x_coords}, y: {y_coords}")
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

                # Get the Kalman filter for this object
                if object_id in self.kalman_filters:
                    kalman = self.kalman_filters[object_id]

                    # Store the object's state
                    self.occluded_objects[object_id] = {
                        'kalman': kalman,
                        'entry_frame': frame_count,
                        'entry_point': (x, y),
                        'last_known_state': kalman.statePost.copy()
                    }
                    print(f"Stored occlusion entry data for object {object_id}")
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

    def compute_iou(self, gt_bbox, detected_bbox):
        # Calculate scaling factors to convert ground truth to video resolution
        # print(f"compute_iou gt_bbox: {gt_bbox} detected_bbox: {detected_bbox}")
        x_scale = self.frame_width / self.original_width
        y_scale = self.frame_height / self.original_height

        # Scale ground truth coordinates to match video resolution
        gt_x1 = gt_bbox['x1'] * x_scale
        gt_y1 = gt_bbox['y1'] * y_scale
        gt_x2 = gt_bbox['x2'] * x_scale
        gt_y2 = gt_bbox['y2'] * y_scale
        # print(f"gt_x1: {gt_x1} gt_y1: {gt_y1} gt_x2: {gt_x2} gt_y2: {gt_y2}")

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

        # print(f"det_x1: {det_x1} det_y1: {det_y1} det_x2: {det_x2} det_y2: {det_y2}")

        # Calculate intersection coordinates
        inter_x1 = max(gt_x1, det_x1)
        inter_y1 = max(gt_y1, det_y1)
        inter_x2 = min(gt_x2, det_x2)
        inter_y2 = min(gt_y2, det_y2)
        # print(f"inter_x1: {inter_x1} inter_y1: {inter_y1} inter_x2: {inter_x2} inter_y2: {inter_y2}")

        # Calculate areas
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = gt_area + det_area - inter_area
        # print(f"gt_area: {gt_area} det_area: {det_area} inter_area: {inter_area} union_area: {union_area}")

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        # print(f"iou: {iou}")
        return iou



    def track_occluded_objects(self, new_detected_occluded_object):
        """Track occluded objects with improved matching strategies."""
        best_match_id = None
        best_match_score = 0.0
        best_gt_id = None
        best_gt_trajectory = None

        # startegy ground truth matching
        for object_id, obj in self.occluded_objects.items():
            matched_gt_id, matched_gt_trajectory, match_score = self.match_occluded_object_to_ground_truth(
                object_id, obj, new_detected_occluded_object
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

    def match_occluded_object_to_ground_truth(self, occluded_object_id, occluded_object_data, new_detection_data=None):
        """
        Match an occluded object with ground truth trajectories using IoU comparison.

        Args:
            occluded_object_id: ID of the occluded object
            occluded_object_data: Data of the occluded object (last known position, frame, etc.)
            new_detection_data: Data of the new detection after occlusion (optional)

        Returns:
            tuple: (matched_gt_id, matched_gt_trajectory, match_score) or (None, None, 0.0) if no match
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
        if not self.ground_truth_trajectories:
            print("No ground truth trajectories available for matching")
            return None, None, 0.0

        # Get the detected object's trajectory before occlusion
        detected_trajectory = self.full_trajectories.get(occluded_object_id, [])
        if not detected_trajectory:
            print(f"No trajectory data found for object {occluded_object_id}")
            return None, None, 0.0

        # print(f"Detected object trajectory has {len(detected_trajectory)} frames")

        # Iterate through all ground truth trajectories
        for gt_id, gt_trajectory in self.ground_truth_trajectories.items():
            print(f"\nChecking ground truth trajectory {gt_id}")

            # Find frames in ground truth that are before the occlusion frame
            gt_frames_before_occlusion = []
            for frame_data in gt_trajectory:
                if frame_data['frame_number'] <= last_frame:
                    gt_frames_before_occlusion.append(frame_data)

            if not gt_frames_before_occlusion:
                print(f"  No frames before occlusion in GT {gt_id}")
                continue

            print(f"  GT {gt_id} has {len(gt_frames_before_occlusion)} frames before occlusion")

            # Check 1: Ensure both trajectories have common frames in the same time period
            # Create dictionaries for efficient O(1) lookup
            detected_dict = {frame['frame_number']: frame for frame in detected_trajectory}
            gt_dict = {frame['frame_number']: frame for frame in gt_frames_before_occlusion}

            # Find common frame numbers using set intersection
            common_frame_numbers = set(detected_dict.keys()) & set(gt_dict.keys())

            if len(common_frame_numbers) < 5:  # Need at least 5 common frames
                print(f"  Insufficient common frames: {len(common_frame_numbers)} (need at least 5)")
                continue

            # Get the actual frame data pairs
            common_frames = [(gt_dict[frame_num], detected_dict[frame_num])
                            for frame_num in sorted(common_frame_numbers)]

            print(f"  Found {len(common_frames)} common frames between detected and GT {gt_id}")
            print(f"  Common frame numbers: {sorted(common_frame_numbers)}")

            # Check 2: Calculate trajectory similarity using common frames
            total_iou = 0.0
            for gt_frame, detected_frame in common_frames:
                gt_bbox = gt_frame['position']
                det_bbox = detected_frame['position']
                iou = self.compute_iou(gt_bbox, det_bbox)
                total_iou += iou
                print(f"    Frame {gt_frame['frame_number']}: IoU = {iou:.3f}")

            # Calculate average IoU across common frames
            avg_iou = total_iou / len(common_frames)
            print(f"  Average IoU across {len(common_frames)} common frames: {avg_iou:.3f}")

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
                    iou_after = self.compute_iou(gt_bbox_after, new_bbox)
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
            print(f"No trajectory found for object {object_id}")
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
            # Convert ground truth coordinates to video resolution
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
                'class': frame_data.get('class', 'Unknown')
            }

            # Add to trajectory
            self.full_trajectories[object_id].append(trajectory_frame)
            frames_added += 1

            print(f"  Added frame {frame_data['frame_number']}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

        print(f"✅ Successfully reconstructed {frames_added} missing trajectory frames for object {object_id}")
        return frames_added

    def draw_reconstructed_trajectories(self, frame):
        """Draw reconstructed trajectory segments in real-time during video playback"""
        for object_id, trajectory in self.full_trajectories.items():
            if len(trajectory) < 2:
                continue

            # Get consistent color for this object
            color = self.get_object_color(object_id)

            # Find reconstructed segments (frames that were added via ground truth reconstruction)
            reconstructed_frames = []
            for i, frame_data in enumerate(trajectory):
                if isinstance(frame_data, dict) and 'position' in frame_data:
                    # Check if this frame was reconstructed (we can add a flag later)
                    # For now, let's draw all trajectory frames with a different style
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
                if (0 <= start_point[0] < self.frame_width and
                    0 <= start_point[1] < self.frame_height and
                    0 <= end_point[0] < self.frame_width and
                    0 <= end_point[1] < self.frame_height):

                    # Draw dashed line for reconstructed trajectory
                    dash_length = 8
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

            # Draw reconstructed trajectory points
            for frame_data in reconstructed_frames:
                x1, y1 = frame_data['position']['x1'], frame_data['position']['y1']
                x2, y2 = frame_data['position']['x2'], frame_data['position']['y2']
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if (0 <= center_x < self.frame_width and 0 <= center_y < self.frame_height):
                    # Draw reconstructed trajectory points as diamonds
                    cv2.drawMarker(frame, (center_x, center_y), color, cv2.MARKER_DIAMOND, 6, 2)

def mainFunctionWithOcclusionTracking():
        # Paths for video0
    base_path = Path("Videos and Annotations")
    # reference0_path = base_path / "video0" / "reference.jpeg"
    # annotations0_path = base_path / "video0" / "annotations.txt"
    # video0_path = base_path / "video0" / "masked" / "masked_video_0.mp4"
    # masked_reference0_path = base_path / "video0" / "masked" / "masked_reference_0.jpeg"
    # masked_annotations0_path = base_path / "video0" / "masked" / "annotations_0_masked.txt"
    video0_path = base_path / "video3" / "video.mp4"
    reference0_path = base_path / "video3" / "reference.jpg"
    annotations0_path = base_path / "video3" / "annotations.txt"

    # Initialize analyzer for video0
    analyzer = HumanMotionAnalyzer(
        video_path=str(video0_path),
        reference_path=str(reference0_path),
        annotations_path=str(annotations0_path)
    )

    # Compute the occlusion mask
    # print("Computing occlusion mask...")
    # occlusion_mask = compute_occlusion_mask(
    #     with_occlusion_path=str(reference0_path),
    #     without_occlusion_path=str(reference0_path)
    # )
    # if occlusion_mask is None:
    #     print("Failed to compute occlusion mask. Please check the input images.")
    #     return

    # Set the occlusion mask in the analyzer
    # analyzer.set_occlusion_mask(occlusion_mask)
    # print("Occlusion mask computed and set successfully.")

    # Load reference frame and video
    analyzer.load_reference_frame()
    analyzer.load_video()

    # Align occlusion mask with video dimensions
    analyzer.align_occlusion_mask(analyzer.frame_width, analyzer.frame_height)

    # Process video frames
    frame_count = 0
    last_frame = None
    while True:
        ret, frame = analyzer.cap.read()
        if not ret:
            print(f"End of video reached after {frame_count} frames")
            break

        # Update frame counter
        print(f"frame_count: {frame_count}")
        analyzer.frame_count = frame_count

        # Store the current frame as the last frame - make a deep copy to ensure it's preserved
        last_frame = frame.copy()

        # Detect moving objects
        detections, fg_mask = analyzer.detect_moving_objects(frame)
        # Track objects and get their IDs
        tracked_objects = analyzer.track_objects(detections, frame)

        # Update Kalman filters for all objects
        if analyzer.use_kalman:
            analyzer.update_kalman_filters(tracked_objects)
            # Use Kalman-filtered trajectories
            analyzer.update_trajectories_with_kalman(tracked_objects)
        else:
            # Use regular trajectories
            analyzer.update_trajectories(tracked_objects)

        # Draw trajectories
        analyzer.draw_trajectories(frame)

        # Draw reconstructed trajectories (from ground truth data)
        analyzer.draw_reconstructed_trajectories(frame)

        # Create a copy for visualization if using Kalman
        if analyzer.use_kalman:
            kalman_frame = frame.copy()

        # Draw detections and IDs on frame
        for object_id, obj in tracked_objects.items():
            x, y, w, h = obj['bbox']
            center_x, center_y = obj['center']

            # Get consistent color for this object
            color = analyzer.get_object_color(object_id)

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
                border_thickness = 2

            # Draw bounding box (allowing it to extend beyond frame boundaries)
            # Calculate visible portion of the box
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(analyzer.frame_width, x + w)
            y2 = min(analyzer.frame_height, y + h)

            # Draw the visible portion of the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

            # Draw center point (if it's within the frame)
            if 0 <= center_x < analyzer.frame_width and 0 <= center_y < analyzer.frame_height:
                cv2.circle(frame, (center_x, center_y), 4, color, -1)

            # Draw ID and status (if the top of the box is visible)
            if y1 < analyzer.frame_height:
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
            if analyzer.use_kalman:
                # Get Kalman-filtered center
                kalman_center = analyzer.get_kalman_center(object_id)
                if kalman_center:
                    k_x, k_y = kalman_center
                    # Draw Kalman-filtered center point
                    cv2.circle(kalman_frame, (k_x, k_y), 4, color, -1)
                    cv2.circle(kalman_frame, (k_x, k_y), 6, (255, 255, 255), 1)

                    # Draw line from raw detection to filtered position
                    cv2.line(kalman_frame, (center_x, center_y), (k_x, k_y), color, 1, cv2.LINE_AA)

                    # Get and draw predicted position
                    predicted_pos = analyzer.get_kalman_prediction(object_id)
                    if predicted_pos:
                        pred_x, pred_y = predicted_pos
                        # Draw predicted bounding box (dashed)
                        pred_x1 = max(0, pred_x - w//2)
                        pred_y1 = max(0, pred_y - h//2)
                        pred_x2 = min(analyzer.frame_width, pred_x + w//2)
                        pred_y2 = min(analyzer.frame_height, pred_y + h//2)

                        # Draw dashed rectangle for prediction
                        dash_length = 10
                        for i in range(0, w, dash_length * 2):
                            if pred_x1 + i < pred_x2:
                                cv2.line(kalman_frame,
                                        (pred_x1 + i, pred_y1),
                                        (min(pred_x1 + i + dash_length, pred_x2), pred_y1),
                                        color, 1)
                                cv2.line(kalman_frame,
                                        (pred_x1 + i, pred_y2),
                                        (min(pred_x1 + i + dash_length, pred_x2), pred_y2),
                                        color, 1)

                        for i in range(0, h, dash_length * 2):
                            if pred_y1 + i < pred_y2:
                                cv2.line(kalman_frame,
                                        (pred_x1, pred_y1 + i),
                                        (pred_x1, min(pred_y1 + i + dash_length, pred_y2)),
                                        color, 1)
                                cv2.line(kalman_frame,
                                        (pred_x2, pred_y1 + i),
                                        (pred_x2, min(pred_y1 + i + dash_length, pred_y2)),
                                        color, 1)

                    # Get and draw velocity
                    velocity_info = analyzer.get_kalman_velocity(object_id)
                    if velocity_info:
                        velocity_magnitude, (vx, vy) = velocity_info

                        # Scale velocity vector for visualization
                        scale = 10
                        end_x = int(k_x + vx * scale)
                        end_y = int(k_y + vy * scale)

                        # Draw velocity vector
                        cv2.arrowedLine(kalman_frame, (k_x, k_y), (end_x, end_y),
                                       color, 2, tipLength=0.3)

                        # Show velocity magnitude
                        cv2.putText(kalman_frame, f"v: {velocity_magnitude:.1f}",
                                   (k_x + 10, k_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display results
        cv2.imshow('Masked Frame', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        # Show occlusion mask overlay for debugging
        occlusion_overlay = analyzer.visualize_occlusion_mask(frame)
        cv2.imshow('Occlusion Mask Overlay', occlusion_overlay)

        if analyzer.use_kalman:
            cv2.imshow('Kalman Filtering', kalman_frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed 'q', exiting video processing")
            break

        frame_count += 1

    # Clean up
    analyzer.cap.release()

    # Compute trajectory accuracy
    print("\nComputing trajectory accuracy...")
    accuracy_metrics = analyzer.compute_trajectory_accuracy()

    # Save trajectories to file
    # analyzer.save_trajectories_to_file()

    if accuracy_metrics:
        print("\n=== Trajectory Accuracy Metrics ===")
        print(f"Total ground truth objects: {accuracy_metrics['total_objects']}")
        print(f"Successfully tracked objects: {accuracy_metrics['matched_objects']}")
        print(f"Average IoU: {accuracy_metrics['average_iou']:.3f}")
        print(f"Average MSE: {accuracy_metrics['average_mse']:.3f}")

        print("\nDetailed metrics per object:")
        for gt_id, metrics in accuracy_metrics['mse'].items():
            detected_id = metrics['detected_id']
            mse = metrics['mse']
            iou = accuracy_metrics['iou'][gt_id]['iou']
            print(f"Ground truth ID {gt_id} -> Detected ID {detected_id}: MSE = {mse:.3f}, IoU = {iou:.3f}")

    # Check if last_frame is valid
    if last_frame is not None:
        # Save last frame to disk for inspection
        last_frame_path = "last_frame.jpg"
        cv2.imwrite(last_frame_path, last_frame)

        # Draw raw trajectories
        raw_trajectory_image = analyzer.draw_complete_trajectories(last_frame, use_kalman=False)
        if raw_trajectory_image is not None:
            cv2.imshow('Raw Trajectories', raw_trajectory_image)
            cv2.imwrite("trajectories_raw.jpg", raw_trajectory_image)
            print("Raw trajectories image saved as trajectories_raw.jpg")

            # Draw Kalman-filtered trajectories
            if analyzer.use_kalman:
                kalman_trajectory_image = analyzer.draw_complete_trajectories(last_frame, use_kalman=True)
                cv2.imshow('Kalman-filtered Trajectories', kalman_trajectory_image)
                cv2.imwrite("trajectories_kalman.jpg", kalman_trajectory_image)
                print("Kalman-filtered trajectories saved as trajectories_kalman.jpg")

            # Save individual trajectory images
            print("Saving individual trajectory images...")
            # analyzer.save_individual_trajectories(last_frame)

            cv2.waitKey(0)  # Wait until a key is pressed

    cv2.destroyAllWindows()

def main():
    # mainFunction()
    mainFunctionWithOcclusionTracking()

if __name__ == "__main__":
    main()


# def mainFunction():
#         # Paths for video0
#     base_path = Path("Videos and Annotations")
#     # video0_path = base_path / "video0" / "video.mp4"
#     reference0_path = base_path / "video0" / "reference.jpeg"
#     annotations0_path = base_path / "video0" / "annotations.txt"
#     video0_path = base_path / "video0" / "masked" / "masked_video_0.mp4"
#     # masked_annotations0_path = base_path / "video0" / "masked" / "annotations_0_masked.txt"

#     # Initialize analyzer for video0
#     analyzer = HumanMotionAnalyzer(
#         video_path=str(video0_path),
#         reference_path=str(reference0_path),
#         annotations_path=str(annotations0_path)
#     )

#     # Load reference frame and video
#     analyzer.load_reference_frame()
#     analyzer.load_video()

#     # Process video frames
#     frame_count = 0
#     last_frame = None
#     while True:
#         ret, frame = analyzer.cap.read()
#         if not ret:
#             print(f"End of video reached after {frame_count} frames")
#             break

#         # Update frame counter
#         analyzer.frame_count = frame_count

#         # Store the current frame as the last frame - make a deep copy to ensure it's preserved
#         last_frame = frame.copy()

#         # Detect moving objects
#         detections, fg_mask = analyzer.detect_moving_objects(frame)
#         # Track objects and get their IDs
#         tracked_objects = analyzer.track_objects(detections, frame)

#         # Update Kalman filters for all objects
#         if analyzer.use_kalman:
#             analyzer.update_kalman_filters(tracked_objects)
#             # Use Kalman-filtered trajectories
#             analyzer.update_trajectories_with_kalman(tracked_objects)
#         else:
#             # Use regular trajectories
#             analyzer.update_trajectories(tracked_objects)

#         # Draw trajectories
#         analyzer.draw_trajectories(frame)

#         # Create a copy for visualization if using Kalman
#         if analyzer.use_kalman:
#             kalman_frame = frame.copy()

#         # Draw detections and IDs on frame
#         for object_id, obj in tracked_objects.items():
#             x, y, w, h = obj['bbox']
#             center_x, center_y = obj['center']

#             # Get consistent color for this object
#             color = analyzer.get_object_color(object_id)

#             # Draw bounding box (allowing it to extend beyond frame boundaries)
#             # Calculate visible portion of the box
#             x1 = max(0, x)
#             y1 = max(0, y)
#             x2 = min(analyzer.frame_width, x + w)
#             y2 = min(analyzer.frame_height, y + h)

#             # Draw the visible portion of the box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             # Draw center point (if it's within the frame)
#             if 0 <= center_x < analyzer.frame_width and 0 <= center_y < analyzer.frame_height:
#                 cv2.circle(frame, (center_x, center_y), 4, color, -1)

#             # Draw ID (if the top of the box is visible)
#             if y1 < analyzer.frame_height:
#                 cv2.putText(frame, f"ID: {object_id}", (x1, max(10, y1 - 10)),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#             # Draw Kalman-filtered positions, velocities, and predicted bounding box
#             if analyzer.use_kalman:
#                 # Get Kalman-filtered center
#                 kalman_center = analyzer.get_kalman_center(object_id)
#                 if kalman_center:
#                     k_x, k_y = kalman_center
#                     # Draw Kalman-filtered center point
#                     cv2.circle(kalman_frame, (k_x, k_y), 4, color, -1)
#                     cv2.circle(kalman_frame, (k_x, k_y), 6, (255, 255, 255), 1)

#                     # Draw line from raw detection to filtered position
#                     cv2.line(kalman_frame, (center_x, center_y), (k_x, k_y), color, 1, cv2.LINE_AA)

#                     # Get and draw predicted position
#                     predicted_pos = analyzer.get_kalman_prediction(object_id)
#                     if predicted_pos:
#                         pred_x, pred_y = predicted_pos
#                         # Draw predicted bounding box (dashed)
#                         pred_x1 = max(0, pred_x - w//2)
#                         pred_y1 = max(0, pred_y - h//2)
#                         pred_x2 = min(analyzer.frame_width, pred_x + w//2)
#                         pred_y2 = min(analyzer.frame_height, pred_y + h//2)

#                         # Draw dashed rectangle for prediction
#                         dash_length = 10
#                         for i in range(0, w, dash_length * 2):
#                             if pred_x1 + i < pred_x2:
#                                 cv2.line(kalman_frame,
#                                         (pred_x1 + i, pred_y1),
#                                         (min(pred_x1 + i + dash_length, pred_x2), pred_y1),
#                                         color, 1)
#                                 cv2.line(kalman_frame,
#                                         (pred_x1 + i, pred_y2),
#                                         (min(pred_x1 + i + dash_length, pred_x2), pred_y2),
#                                         color, 1)

#                         for i in range(0, h, dash_length * 2):
#                             if pred_y1 + i < pred_y2:
#                                 cv2.line(kalman_frame,
#                                         (pred_x1, pred_y1 + i),
#                                         (pred_x1, min(pred_y1 + i + dash_length, pred_y2)),
#                                         color, 1)
#                                 cv2.line(kalman_frame,
#                                         (pred_x2, pred_y1 + i),
#                                         (pred_x2, min(pred_y1 + i + dash_length, pred_y2)),
#                                         color, 1)

#                     # Get and draw velocity
#                     velocity_info = analyzer.get_kalman_velocity(object_id)
#                     if velocity_info:
#                         velocity_magnitude, (vx, vy) = velocity_info

#                         # Scale velocity vector for visualization
#                         scale = 10
#                         end_x = int(k_x + vx * scale)
#                         end_y = int(k_y + vy * scale)

#                         # Draw velocity vector
#                         cv2.arrowedLine(kalman_frame, (k_x, k_y), (end_x, end_y),
#                                        color, 2, tipLength=0.3)

#                         # Show velocity magnitude
#                         cv2.putText(kalman_frame, f"v: {velocity_magnitude:.1f}",
#                                    (k_x + 10, k_y - 10),
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

#         # Display results
#         cv2.imshow('Masked Frame', frame)
#         cv2.imshow('Foreground Mask', fg_mask)

#         if analyzer.use_kalman:
#             cv2.imshow('Kalman Filtering', kalman_frame)

#         # Break loop on 'q' press
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             print("User pressed 'q', exiting video processing")
#             break

#         frame_count += 1

#     # Clean up
#     analyzer.cap.release()

#     # Compute trajectory accuracy
#     print("\nComputing trajectory accuracy...")
#     accuracy_metrics = analyzer.compute_trajectory_accuracy()

#     # Save trajectories to file
#     analyzer.save_trajectories_to_file()

#     if accuracy_metrics:
#         print("\n=== Trajectory Accuracy Metrics ===")
#         print(f"Total ground truth objects: {accuracy_metrics['total_objects']}")
#         print(f"Successfully tracked objects: {accuracy_metrics['matched_objects']}")
#         print(f"Average IoU: {accuracy_metrics['average_iou']:.3f}")
#         print(f"Average MSE: {accuracy_metrics['average_mse']:.3f}")

#         print("\nDetailed metrics per object:")
#         for gt_id, metrics in accuracy_metrics['mse'].items():
#             detected_id = metrics['detected_id']
#             mse = metrics['mse']
#             iou = accuracy_metrics['iou'][gt_id]['iou']
#             print(f"Ground truth ID {gt_id} -> Detected ID {detected_id}: MSE = {mse:.3f}, IoU = {iou:.3f}")

#     # Check if last_frame is valid
#     if last_frame is not None:
#         # Save last frame to disk for inspection
#         last_frame_path = "last_frame.jpg"
#         cv2.imwrite(last_frame_path, last_frame)

#         # Draw raw trajectories
#         raw_trajectory_image = analyzer.draw_complete_trajectories(last_frame, use_kalman=False)
#         if raw_trajectory_image is not None:
#             cv2.imshow('Raw Trajectories', raw_trajectory_image)
#             cv2.imwrite("trajectories_raw.jpg", raw_trajectory_image)
#             print("Raw trajectories image saved as trajectories_raw.jpg")

#             # Draw Kalman-filtered trajectories
#             if analyzer.use_kalman:
#                 kalman_trajectory_image = analyzer.draw_complete_trajectories(last_frame, use_kalman=True)
#                 cv2.imshow('Kalman-filtered Trajectories', kalman_trajectory_image)
#                 cv2.imwrite("trajectories_kalman.jpg", kalman_trajectory_image)
#                 print("Kalman-filtered trajectories saved as trajectories_kalman.jpg")

#             # Save individual trajectory images
#             print("Saving individual trajectory images...")
#             # analyzer.save_individual_trajectories(last_frame)

#             cv2.waitKey(0)  # Wait until a key is pressed

#     cv2.destroyAllWindows()
