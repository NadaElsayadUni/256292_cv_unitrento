"""
Annotation file reading utilities
"""

import numpy as np
from collections import defaultdict
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import random
import networkx as nx
from ..config.settings import *


def ensure_output_directory():
    """Ensure the output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_TRAJECTORIES_DIR, exist_ok=True)


def read_annotations_file(file_path):
    """
    Read the annotations file from video0 directory.
    Each line format: ObjectID X1 Y1 X2 Y2 FrameNum Meta1 Meta2 Meta3 "Class"
    Example: 5 397 1270 448 1354 1029 0 0 1 "Biker"

    For masked annotations, coordinates can be "X" to indicate occlusion.

    Parameters:
    -----------
    file_path : str
        Path to the annotations file

    Returns:
    --------
    tuple
        (trajectories, class_counts, frame_distribution)
    """
    trajectories = defaultdict(list)
    class_counts = defaultdict(int)
    frame_distribution = defaultdict(lambda: defaultdict(int))  # frame -> class -> count

    def safe_float(value):
        """Safely convert value to float, returning the original string if conversion fails"""
        try:
            return float(value)
        except ValueError:
            return value  # Return the original string (e.g., "X")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Split the line into components
                components = line.strip().split()

                if len(components) >= 10:  # Ensure we have all required fields
                    object_id = int(components[0])
                    frame_num = int(components[5])
                    class_name = components[9].strip('"')

                    frame_data = {
                        'frame_number': frame_num,
                        'position': {
                            'x1': safe_float(components[1]),
                            'y1': safe_float(components[2]),
                            'x2': safe_float(components[3]),
                            'y2': safe_float(components[4])
                        },
                        'metadata': [int(components[6]), int(components[7]), int(components[8])],
                        'class': class_name
                    }
                    trajectories[object_id].append(frame_data)
                    class_counts[class_name] += 1
                    frame_distribution[frame_num][class_name] += 1
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None, None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, None, None

    # Sort each trajectory by frame number
    for obj_id in trajectories:
        trajectories[obj_id].sort(key=lambda x: x['frame_number'])

    return trajectories, class_counts, frame_distribution


def generate_random_color():
    """Generate a random color in RGB format"""
    return (random.random(), random.random(), random.random())


def visualize_trajectories(trajectories, reference_image_path, output_path):
    """
    Visualize trajectories on the reference image and save individual trajectories

    Parameters:
    -----------
    trajectories : dict
        Dictionary of trajectories
    reference_image_path : str
        Path to the reference image
    output_path : str
        Path to save the visualization
    """
    ensure_output_directory()

    # Create actual_trajectories directory if it doesn't exist
    trajectories_dir = os.path.join(OUTPUT_DIR, "actual_trajectories")
    if not os.path.exists(trajectories_dir):
        os.makedirs(trajectories_dir)

    # Read the reference image
    img = cv2.imread(reference_image_path)
    if img is None:
        print(f"Error: Could not read reference image at {reference_image_path}")
        return

    # Convert to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Generate random colors for each trajectory
    trajectory_colors = {obj_id: generate_random_color() for obj_id in trajectories.keys()}

    # Create main visualization
    plt.figure(figsize=(15, 10))
    plt.imshow(img)

    # Plot each trajectory
    for obj_id, trajectory in trajectories.items():
        # Get center points, filtering out occluded frames
        x_coords = []
        y_coords = []
        for frame in trajectory:
            # Skip frames where coordinates are "X" (occluded)
            if (isinstance(frame['position']['x1'], str) or
                isinstance(frame['position']['y1'], str) or
                isinstance(frame['position']['x2'], str) or
                isinstance(frame['position']['y2'], str)):
                continue

            x_center = (frame['position']['x1'] + frame['position']['x2']) / 2
            y_center = (frame['position']['y1'] + frame['position']['y2']) / 2
            x_coords.append(x_center)
            y_coords.append(y_center)

        # Skip if no valid coordinates
        if len(x_coords) == 0:
            continue

        # Get color for this trajectory
        color = trajectory_colors[obj_id]

        # Plot the trajectory
        plt.plot(x_coords, y_coords, '-', color=color, alpha=0.7, linewidth=2)

        # Plot start and end points with IDs
        plt.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=8)
        plt.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=8)

        # Add ID labels at start and end
        plt.text(x_coords[0], y_coords[0], f'ID:{obj_id}', color=color,
                fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        plt.text(x_coords[-1], y_coords[-1], f'ID:{obj_id}', color=color,
                fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # Create individual trajectory visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.plot(x_coords, y_coords, '-', color=color, alpha=0.7, linewidth=2)
        plt.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=8)
        plt.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=8)
        plt.text(x_coords[0], y_coords[0], f'ID:{obj_id}', color=color,
                fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        plt.text(x_coords[-1], y_coords[-1], f'ID:{obj_id}', color=color,
                fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        plt.title(f'Trajectory of Object {obj_id} ({trajectory[0]["class"]})')

        # Save individual trajectory
        individual_path = os.path.join(trajectories_dir, f'trajectory_{obj_id}.jpg')
        plt.savefig(individual_path, bbox_inches='tight', dpi=300)
        plt.close()

    # Add title to main visualization
    plt.title('Object Trajectories with IDs')

    # Save the main visualization
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Main visualization saved to {output_path}")
    print(f"Individual trajectories saved to {trajectories_dir}/")

    # Show the main plot
    plt.show()


def analyze_trajectories(trajectories, class_counts, frame_distribution):
    """
    Analyze the trajectories and print summary statistics

    Parameters:
    -----------
    trajectories : dict
        Dictionary of trajectories
    class_counts : dict
        Dictionary of class counts
    frame_distribution : dict
        Dictionary of frame distributions

    Returns:
    --------
    dict
        Dictionary containing trajectory statistics including directions
    """
    if not trajectories:
        print("No trajectories to analyze")
        return {}

    print("\n=== Dataset Statistics ===")
    print(f"Total number of tracked objects: {len(trajectories)}")

    # Count unique objects per class
    unique_objects = defaultdict(int)
    for obj_id, trajectory in trajectories.items():
        class_name = trajectory[0]['class']
        unique_objects[class_name] += 1

    print("\nUnique objects per class:")
    for class_name, count in unique_objects.items():
        print(f"  {class_name}: {count} unique objects")

    print("\nTotal detections per class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} detections")

    # Analyze frame distribution
    frames = sorted(frame_distribution.keys())
    if frames:
        print(f"\nFrame range: {frames[0]} to {frames[-1]}")
        print(f"Total number of frames: {len(frames)}")

        # Calculate average objects per frame
        total_objects_per_frame = []
        for frame in frames:
            total = sum(frame_distribution[frame].values())
            total_objects_per_frame.append(total)

        avg_objects = np.mean(total_objects_per_frame)
        max_objects = max(total_objects_per_frame)
        print(f"\nAverage objects per frame: {avg_objects:.2f}")
        print(f"Maximum objects in a single frame: {max_objects}")

        # Find frame with most objects
        max_frame = frames[np.argmax(total_objects_per_frame)]
        print(f"\nFrame with most objects (Frame {max_frame}):")
        for class_name, count in frame_distribution[max_frame].items():
            print(f"  {class_name}: {count}")

    print("\n=== Detailed Object Trajectories ===")
    # Calculate statistics for each object
    trajectory_stats = {}
    for obj_id, trajectory in trajectories.items():
        frames = [frame['frame_number'] for frame in trajectory]
        start_frame = min(frames)
        end_frame = max(frames)
        duration = end_frame - start_frame + 1

        # Calculate center points, filtering out occluded frames
        x_coords = []
        y_coords = []
        for frame in trajectory:
            # Skip frames where coordinates are "X" (occluded)
            if (isinstance(frame['position']['x1'], str) or
                isinstance(frame['position']['y1'], str) or
                isinstance(frame['position']['x2'], str) or
                isinstance(frame['position']['y2'], str)):
                continue

            x_center = (frame['position']['x1'] + frame['position']['x2']) / 2
            y_center = (frame['position']['y1'] + frame['position']['y2']) / 2
            x_coords.append(x_center)
            y_coords.append(y_center)

        # Skip if no valid coordinates
        if len(x_coords) == 0:
            print(f"\nObject {obj_id} ({trajectory[0]['class']}):")
            print(f"  Skipped - all frames are occluded")
            continue

        # Calculate total distance
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        distances = np.sqrt(dx**2 + dy**2)
        total_distance = np.sum(distances)

        # Calculate average speed (pixels per frame)
        avg_speed = total_distance / duration if duration > 0 else 0

        # Calculate direction of movement
        if len(x_coords) > 1:
            dx_total = x_coords[-1] - x_coords[0]
            dy_total = y_coords[-1] - y_coords[0]
            direction = np.arctan2(dy_total, dx_total) * 180 / np.pi
        else:
            direction = 0

        # Store statistics
        trajectory_stats[obj_id] = {
            'direction': direction,
            'start_pos': (x_coords[0], y_coords[0]),
            'end_pos': (x_coords[-1], y_coords[-1]),
            'duration': duration,
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'class': trajectory[0]['class']
        }

        print(f"\nObject {obj_id} ({trajectory[0]['class']}):")
        print(f"  Duration: {duration} frames")
        print(f"  Start frame: {start_frame}")
        print(f"  End frame: {end_frame}")
        print(f"  Total distance: {total_distance:.2f} pixels")
        print(f"  Average speed: {avg_speed:.2f} pixels/frame")
        print(f"  Start position: ({x_coords[0]:.1f}, {y_coords[0]:.1f})")
        print(f"  End position: ({x_coords[-1]:.1f}, {y_coords[-1]:.1f})")
        print(f"  Direction of movement: {direction:.1f}°")

    return trajectory_stats


def analyze_path_patterns(trajectories, reference_image_path, trajectory_stats):
    """
    Analyze the most frequent paths by determining entry and exit points
    of trajectories relative to the image boundaries.

    Parameters:
    -----------
    trajectories : dict
        Dictionary of trajectories
    reference_image_path : str
        Path to the reference image to get dimensions
    trajectory_stats : dict
        Dictionary containing trajectory statistics including directions

    Returns:
    --------
    dict
        Dictionary containing path pattern statistics
    """
    # Get image dimensions from reference image
    img = cv2.imread(reference_image_path)
    if img is None:
        print(f"Error: Could not read reference image at {reference_image_path}")
        return {}, {}

    height, width = img.shape[:2]

    # Define image boundaries using actual dimensions
    LEFT = 0
    RIGHT = width
    TOP = 0
    BOTTOM = height

    # Define margin for considering a point as "on the boundary"
    # Use 20% of the smaller dimension as margin
    MARGIN = min(width, height) * 0.25

    def get_boundary_position(x, y):
        """Determine which boundary a point is closest to"""
        if x <= LEFT + MARGIN:
            return "LEFT"
        elif x >= RIGHT - MARGIN:
            return "RIGHT"
        elif y <= TOP + MARGIN:
            return "TOP"
        elif y >= BOTTOM - MARGIN:
            return "BOTTOM"
        return "CENTER"

    def find_nearest_path(stats, path_patterns):
        """Find the nearest path pattern based on position and direction"""
        # Define direction ranges for each boundary
        direction_ranges = {
            "LEFT": (135, 225),     # Left side (includes angles around 180°)
            "RIGHT": (-45, 45),     # Right side
            "TOP": (45, 135),       # Top side
            "BOTTOM": (225, 315)    # Bottom side
        }

        # Get start and end positions
        start_x, start_y = stats['start_pos']
        end_x, end_y = stats['end_pos']
        direction = stats['direction']

        # Determine likely entry and exit points based on direction
        entry_point = None
        exit_point = None

        # Normalize direction to 0-360 range
        direction = direction % 360

        # Determine entry point based on start position
        if start_x <= LEFT + MARGIN:
            entry_point = "LEFT"
        elif start_x >= RIGHT - MARGIN:
            entry_point = "RIGHT"
        elif start_y <= TOP + MARGIN:
            entry_point = "TOP"
        elif start_y >= BOTTOM - MARGIN:
            entry_point = "BOTTOM"

        # Determine exit point based on end position
        if end_x <= LEFT + MARGIN:
            exit_point = "LEFT"
        elif end_x >= RIGHT - MARGIN:
            exit_point = "RIGHT"
        elif end_y <= TOP + MARGIN:
            exit_point = "TOP"
        elif end_y >= BOTTOM - MARGIN:
            exit_point = "BOTTOM"

        # If we couldn't determine entry/exit points, use direction
        if not entry_point:
            for boundary, (min_angle, max_angle) in direction_ranges.items():
                if min_angle <= direction <= max_angle:
                    entry_point = boundary
                    break

        if not exit_point:
            # Use opposite direction for exit point
            opposite_direction = (direction + 180) % 360
            for boundary, (min_angle, max_angle) in direction_ranges.items():
                if min_angle <= opposite_direction <= max_angle:
                    exit_point = boundary
                    break

        if entry_point == "LEFT" and exit_point == "LEFT":
            entry_point = "BOTTOM"
            exit_point = "LEFT"
        elif entry_point == "RIGHT" and exit_point == "RIGHT":
            entry_point = "LEFT"
            exit_point = "RIGHT"
        elif entry_point == "TOP" and exit_point == "TOP":
            entry_point = "BOTTOM"
            exit_point = "TOP"
        elif entry_point == "BOTTOM" and exit_point == "BOTTOM":
            entry_point = "BOTTOM"
            exit_point = "TOP"

        # Ensure we have valid entry and exit points
        # Default to most common path if we still have no valid points
        if exit_point == None:
            if entry_point == "LEFT":
                exit_point = "RIGHT"
            elif entry_point == "RIGHT":
                exit_point = "LEFT"
            elif entry_point == "TOP":
                exit_point = "BOTTOM"
            elif entry_point == "BOTTOM" :
                exit_point = "TOP"

        if entry_point == None:
            if exit_point == "LEFT":
                entry_point = "RIGHT"
            elif exit_point == "RIGHT":
                entry_point = "LEFT"
            elif exit_point == "TOP":
                entry_point = "BOTTOM"
            elif exit_point == "BOTTOM":
                entry_point = "TOP"

        return f"{entry_point} to {exit_point}"

    # Analyze each trajectory
    path_patterns = defaultdict(int)
    detailed_patterns = defaultdict(list)

    # Statistics tracking
    total_objects = len(trajectories)
    analyzed_objects = 0
    excluded_objects = {
        'short_trajectory': [],
        'center_entry': [],
        'center_exit': [],
        'center_both': []
    }

    # First pass: collect all valid paths
    for obj_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            excluded_objects['short_trajectory'].append(obj_id)
            continue

        # Get start and end points, checking for occluded frames
        start_frame = trajectory[0]
        end_frame = trajectory[-1]

        # Check if start or end frames are occluded
        if (isinstance(start_frame['position']['x1'], str) or
            isinstance(start_frame['position']['y1'], str) or
            isinstance(start_frame['position']['x2'], str) or
            isinstance(start_frame['position']['y2'], str) or
            isinstance(end_frame['position']['x1'], str) or
            isinstance(end_frame['position']['y1'], str) or
            isinstance(end_frame['position']['x2'], str) or
            isinstance(end_frame['position']['y2'], str)):
            # Skip objects with occluded start or end frames
            excluded_objects['short_trajectory'].append(obj_id)
            continue

        start_x = (start_frame['position']['x1'] + start_frame['position']['x2']) / 2
        start_y = (start_frame['position']['y1'] + start_frame['position']['y2']) / 2
        end_x = (end_frame['position']['x1'] + end_frame['position']['x2']) / 2
        end_y = (end_frame['position']['y1'] + end_frame['position']['y2']) / 2

        # Determine entry and exit points
        entry_point = get_boundary_position(start_x, start_y)
        exit_point = get_boundary_position(end_x, end_y)

        # Track why objects are excluded
        if entry_point == "CENTER" and exit_point == "CENTER":
            excluded_objects['center_both'].append(obj_id)
            continue
        elif entry_point == "CENTER":
            excluded_objects['center_entry'].append(obj_id)
            continue
        elif exit_point == "CENTER":
            excluded_objects['center_exit'].append(obj_id)
            continue

        if entry_point == "LEFT" and exit_point == "LEFT":
            entry_point = "BOTTOM"
            exit_point = "LEFT"
        elif entry_point == "RIGHT" and exit_point == "RIGHT":
            entry_point = "LEFT"
            exit_point = "RIGHT"
        elif entry_point == "TOP" and exit_point == "TOP":
            entry_point = "BOTTOM"
            exit_point = "TOP"
        elif entry_point == "BOTTOM" and exit_point == "BOTTOM":
            entry_point = "BOTTOM"
            exit_point = "TOP"

        # Create path pattern
        pattern = f"{entry_point} to {exit_point}"
        path_patterns[pattern] += 1
        detailed_patterns[pattern].append({
            'object_id': obj_id,
            'class': trajectory[0]['class'],
            'start_pos': (start_x, start_y),
            'end_pos': (end_x, end_y),
            'entry_point': entry_point,
            'exit_point': exit_point,
            'start_frame': start_frame['frame_number'],
            'end_frame': end_frame['frame_number']
        })
        analyzed_objects += 1

    # Second pass: assign excluded objects to nearest paths
    for obj_id in excluded_objects['short_trajectory'] + excluded_objects['center_entry'] + \
                  excluded_objects['center_exit'] + excluded_objects['center_both']:
        if obj_id not in trajectory_stats:
            continue

        stats = trajectory_stats[obj_id]

        # Find nearest path
        if obj_id not in excluded_objects['center_both']:
            pattern = find_nearest_path(stats, path_patterns)
        else:
            pattern = "CENTER to CENTER"

        # Add to path patterns
        path_patterns[pattern] += 1
        detailed_patterns[pattern].append({
            'object_id': obj_id,
            'class': stats['class'],
            'start_pos': stats['start_pos'],
            'end_pos': stats['end_pos'],
            'entry_point': pattern.split(" to ")[0],
            'exit_point': pattern.split(" to ")[1],
            'direction': stats['direction'],
            'assigned': True  # Mark as assigned
        })
        analyzed_objects += 1

    # Print detailed statistics
    print(f"\nPath Analysis Statistics:")
    print(f"Total objects: {total_objects}")
    print(f"Analyzed objects: {analyzed_objects} ({analyzed_objects/total_objects*100:.1f}%)")
    print(f"Excluded objects: {total_objects - analyzed_objects} ({(total_objects - analyzed_objects)/total_objects*100:.1f}%)")

    if excluded_objects['short_trajectory']:
        print(f"\nObjects with short trajectories (< 2 frames): {len(excluded_objects['short_trajectory'])}")
        print("Example objects:", excluded_objects['short_trajectory'])

    if excluded_objects['center_both']:
        print(f"\nObjects with center entry and exit: {len(excluded_objects['center_both'])}")
        print("Example objects:", excluded_objects['center_both'])

    if excluded_objects['center_entry']:
        print(f"\nObjects with center entry: {len(excluded_objects['center_entry'])}")
        print("Example objects:", excluded_objects['center_entry'])

    if excluded_objects['center_exit']:
        print(f"\nObjects with center exit: {len(excluded_objects['center_exit'])}")
        print("Example objects:", excluded_objects['center_exit'])

    return path_patterns, detailed_patterns


def visualize_path_patterns(path_patterns, output_path=PATH_PATTERNS_GRAPH):
    """
    Create a graph visualization of path patterns showing all possible paths between boundaries.

    Parameters:
    -----------
    path_patterns : dict
        Dictionary containing path patterns and their frequencies
    output_path : str
        Path to save the visualization
    """
    ensure_output_directory()

    # Create a directed graph
    G = nx.DiGraph()

    # Define all possible boundaries
    boundaries = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT']

    # Add all possible edges with initial weight 0
    for source in boundaries:
        for target in boundaries:
            if source != target:  # Don't connect a boundary to itself
                G.add_edge(source, target, weight=0)

    # Update weights based on actual path patterns
    for pattern, count in path_patterns.items():
        if pattern != "CENTER to CENTER":  # Skip center-to-center paths
            source, target = pattern.split(" to ")
            G[source][target]['weight'] = count

    # Create the plot
    plt.figure(figsize=(15, 10))

    # Define custom node positions (TOP at top, LEFT at left, etc.)
    pos = {
        'TOP': (0.5, 1.0),
        'RIGHT': (1.0, 0.5),
        'BOTTOM': (0.5, 0.0),
        'LEFT': (0.0, 0.5)
    }

    # Draw edges with varying thickness based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + 3 * (w / max_weight) for w in edge_weights]

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', alpha=0.8)

    # Draw edges with curved paths and adjusted arrow positions
    for i, (u, v) in enumerate(G.edges()):
        # Calculate curve direction based on node positions
        if (u == 'TOP' and v == 'LEFT') or (u == 'LEFT' and v == 'BOTTOM') or \
           (u == 'BOTTOM' and v == 'RIGHT') or (u == 'RIGHT' and v == 'TOP') or \
           (u == 'TOP' and v == 'RIGHT') or (u == 'RIGHT' and v == 'BOTTOM') or \
           (u == 'BOTTOM' and v == 'LEFT') or (u == 'LEFT' and v == 'TOP'):
            rad = 0.2
        else:
            rad = -0.2

        # Draw forward edge
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=[edge_widths[i]],
                             arrows=True, arrowsize=15, connectionstyle=f'arc3,rad={rad}',
                             arrowstyle='-|>', min_source_margin=15, min_target_margin=15)

        # Draw reverse edge
        reverse_weight = G[v][u]['weight']
        reverse_width = 2 + 3 * (reverse_weight / max_weight)
        nx.draw_networkx_edges(G, pos, edgelist=[(v, u)], width=[reverse_width],
                             arrows=True, arrowsize=15, connectionstyle=f'arc3,rad={-rad}',
                             arrowstyle='-', min_source_margin=15, min_target_margin=15)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Add edge labels showing the count
    edge_labels = {}
    for u, v in G.edges():
        # Get the count for this direction
        forward_count = G[u][v]['weight']
        # Get the count for the reverse direction
        reverse_count = G[v][u]['weight']

        # Create a label showing both directions
        edge_labels[(u, v)] = f"{reverse_count} ←\n→ {forward_count}"

    # Draw edge labels with adjusted positions
    for (u, v), label in edge_labels.items():
        # Calculate midpoint of the edge
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Adjust position based on edge direction
        if u == 'TOP' and v == 'BOTTOM':
            mid_x += 0.1  # Move right
        elif u == 'BOTTOM' and v == 'TOP':
            mid_x -= 0.1  # Move left
        elif u == 'LEFT' and v == 'RIGHT':
            mid_y += 0.1  # Move up
        elif u == 'RIGHT' and v == 'LEFT':
            mid_y -= 0.1  # Move down

        plt.text(mid_x, mid_y, label, fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.title("All Possible Path Patterns with Counts", fontsize=16, pad=20)
    plt.axis('off')

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Path patterns graph saved to {output_path}")
    plt.show()