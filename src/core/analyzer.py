"""
Main analyzer module for human motion analysis

This module contains the HumanMotionAnalyzer class that orchestrates
all the analysis components including detection, tracking, Kalman filtering,
occlusion handling, and trajectory analysis.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

from .detection import ObjectDetector
from .tracking import ObjectTracker
from .kalman_filter import KalmanFilterManager
from .occlusion import OcclusionHandler
from ..utils.visualization import Visualizer
from ..utils.metrics import compute_trajectory_accuracy
from ..utils.file_utils import save_trajectories_to_file, save_final_analysis_results, save_ground_truth_trajectories
from ..data.annotations import read_annotations_file
from ..config.settings import *


class HumanMotionAnalyzer:
    """
    Main analyzer class for human motion analysis in videos.

    This class orchestrates all the analysis components including object detection,
    tracking, Kalman filtering, occlusion handling, and trajectory analysis.

    This is based on the original HumanMotionAnalyzer from final_one.py but
    refactored to use the modular architecture.
    """

    def __init__(self, video_path, reference_path, annotations_path):
        """
        Initialize the human motion analyzer.

        Parameters:
        -----------
        video_path : str
            Path to the input video file
        reference_path : str
            Path to the reference/background image
        annotations_path : str
            Path to the ground truth annotations file
        """
        self.video_path = video_path
        self.reference_path = reference_path
        self.annotations_path = annotations_path

        # Video and frame management
        self.cap = None
        self.reference_frame = None
        self.frame_count = 0
        self.frame_width = None
        self.frame_height = None

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
            # print(f"Determined original resolution from ground truth: {self.original_width}x{self.original_height}")
            # print(f"Loaded {len(self.ground_truth_trajectories)} complete ground truth trajectories for reconstruction")
        else:
            # Default values if ground truth can't be read
            self.original_width = 1920
            self.original_height = 1080
            # print("Warning: Could not read ground truth, using default resolution")

        # Initialize analysis components
        self.detector = ObjectDetector()
        self.tracker = None
        self.kalman_manager = KalmanFilterManager()
        self.occlusion_handler = OcclusionHandler()
        self.visualizer = Visualizer()

        # Store ground truth for analysis
        self.ground_truth_trajectories = self.ground_truth_trajectories

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

        # Initialize tracker with video dimensions and original dimensions for scaling
        self.tracker = ObjectTracker(self.frame_width, self.frame_height, self.original_width, self.original_height)

        return self.cap

    def set_occlusion_mask(self, mask):
        """Set the occlusion mask for the analyzer"""
        self.occlusion_handler.set_occlusion_mask(mask)
        print(f"Occlusion mask set with shape: {mask.shape if mask is not None else 'None'}")

    def align_occlusion_mask(self, video_width, video_height):
        """Align the occlusion mask with the video dimensions"""
        self.occlusion_handler.align_occlusion_mask(video_width, video_height)

    def process_frame(self, frame):
        """
        Process a single frame through the complete analysis pipeline.

        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame to process

        Returns:
        --------
        dict
            Dictionary containing processing results
        """
        # Update frame counter
        self.frame_count += 1

        # Detect moving objects
        # detections, fg_mask = self.detector.detect_moving_objects(frame)
        detections, fg_mask, non_shadow_mask = self.detector.detect_moving_objects_with_shadow_suppression(frame)

        # Ensure tracker is initialized
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized. Call load_video() first.")

        # Update tracker frame count
        self.tracker.set_frame_count(self.frame_count)

        # Track objects
        tracked_objects = self.tracker.track_objects(
            detections,
            frame,
            self.occlusion_handler,
            self.ground_truth_trajectories
        )

        # Update Kalman filters
        if self.kalman_manager.use_kalman:
            self.kalman_manager.update_kalman_filters(tracked_objects)
            # Use Kalman-filtered trajectories
            self.tracker.update_trajectories_with_kalman(
                tracked_objects,
                self.kalman_manager,
                self.occlusion_handler
            )
        else:
            # Use regular trajectories
            self.tracker.update_trajectories(tracked_objects)

        return {
            'detections': detections,
            'tracked_objects': tracked_objects,
            'fg_mask': fg_mask,
            'non_shadow_mask': non_shadow_mask,
            'frame_count': self.frame_count
        }

    def draw_results(self, frame, results, use_kalman=False):
        """
        Draw analysis results on the frame.

        Parameters:
        -----------
        frame : numpy.ndarray
            Frame to draw on
        results : dict
            Results from process_frame()
        use_kalman : bool
            Whether to show Kalman filtering results

        Returns:
        --------
        numpy.ndarray
            Frame with visualizations
        """
        tracked_objects = results['tracked_objects']

        # Ensure tracker is initialized
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized. Call load_video() first.")

        # Draw trajectories using visualizer
        print(f"Drawing results")
        # self.visualizer.draw_trajectories(frame, self.tracker.trajectories, self.frame_width, self.frame_height)

        # Draw reconstructed trajectories (from ground truth data)
        # self.visualizer.draw_reconstructed_trajectories(frame, self.tracker.full_trajectories, self.frame_width, self.frame_height)

        # Draw detections and IDs using visualizer
        self.visualizer.draw_objects(
            frame, tracked_objects, self.frame_width, self.frame_height,
            self.occlusion_handler, self.kalman_manager
        )

        return frame

    def compute_accuracy(self):
        """Compute trajectory accuracy against ground truth"""
        print("\nComputing trajectory accuracy...")

        # Ensure tracker is initialized
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized. Call load_video() first.")

        accuracy_metrics = compute_trajectory_accuracy(
            self.tracker.full_trajectories,
            self.annotations_path,
            self.frame_width,
            self.frame_height,
            self.original_width,
            self.original_height
        )
        return accuracy_metrics

    def save_results(self, last_frame=None):
        """Save analysis results and visualizations"""
        # Ensure tracker is initialized
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized. Call load_video() first.")

        # Save trajectories to file
        save_trajectories_to_file(self.tracker.full_trajectories)

        # Save last frame if provided
        if last_frame is not None:
            cv2.imwrite(LAST_FRAME_IMAGE, last_frame)

            # Draw Kalman-filtered trajectories
            if self.kalman_manager.use_kalman:
                kalman_centers = self.kalman_manager.kalman_centers
                kalman_trajectory_image = self.visualizer.draw_complete_trajectories(
                    last_frame, self.tracker.full_trajectories, use_kalman=True, kalman_centers=kalman_centers
                )
                cv2.imwrite(TRAJECTORIES_KALMAN_IMAGE, kalman_trajectory_image)
                print(f"Kalman-filtered trajectories saved as {TRAJECTORIES_KALMAN_IMAGE}")
            else:
                # Draw and save trajectory visualizations using visualizer
                raw_trajectory_image = self.visualizer.draw_complete_trajectories(
                    last_frame, self.tracker.full_trajectories, use_kalman=False
                )
                if raw_trajectory_image is not None:
                    cv2.imwrite(TRAJECTORIES_RAW_IMAGE, raw_trajectory_image)
                    print(f"Raw trajectories image saved as {TRAJECTORIES_RAW_IMAGE}")


            # Save individual trajectory images
            # self.visualizer.save_individual_trajectories(last_frame, self.tracker.full_trajectories, INDIVIDUAL_TRAJECTORIES_DIR)

            # Save ground truth trajectories for comparison
            # if self.ground_truth_trajectories:
            #     save_ground_truth_trajectories(
            #         last_frame=last_frame,
            #         ground_truth_trajectories=self.ground_truth_trajectories,
            #         output_dir=GROUND_TRUTH_TRAJECTORIES_DIR,
            #         frame_width=self.frame_width,
            #         frame_height=self.frame_height,
            #         original_width=self.original_width,
            #         original_height=self.original_height
            #     )

    def run_analysis(self, show_visualization=True, save_results=True):
        """
        Run the complete analysis pipeline on the video.

        Parameters:
        -----------
        show_visualization : bool
            Whether to show real-time visualization
        save_results : bool
            Whether to save results and visualizations
        """
        # Load video and reference
        self.load_reference_frame()
        self.load_video()

        # Align occlusion mask if available
        if self.occlusion_handler.occlusion_mask is not None:
            self.align_occlusion_mask(self.frame_width, self.frame_height)

        # Process video frames
        frame_count = 0
        last_frame = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"End of video reached after {frame_count} frames")
                break

            # Store the current frame as the last frame
            last_frame = frame.copy()

            # Process frame
            results = self.process_frame(frame)

            # Draw results
            frame = self.draw_results(frame, results, use_kalman=self.kalman_manager.use_kalman)

            # Display results
            if show_visualization:
                cv2.imshow('Analysis Results', frame)
                cv2.imshow('Foreground Mask', results['fg_mask'])
                cv2.imshow('Shadow Mask (Non-Shadow Areas)', results['non_shadow_mask'])

                # Show occlusion mask overlay only if occlusion tracking is enabled
                if self.occlusion_handler.occlusion_mask is not None:
                    occlusion_overlay = self.occlusion_handler.visualize_occlusion_mask(frame)
                    cv2.imshow('Occlusion Mask Overlay', occlusion_overlay)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User pressed 'q', exiting video processing")
                    break

            frame_count += 1

        # Clean up
        self.cap.release()

        # Compute accuracy
        accuracy_metrics = self.compute_accuracy()

        # Save results
        if save_results:
            self.save_results(last_frame)

        # Display final results
        if accuracy_metrics:
            print("\n=== Final Analysis Results ===")
            print(f"Total ground truth objects: {accuracy_metrics['total_objects']}")
            print(f"Successfully tracked objects: {accuracy_metrics['matched_objects']}")
            print(f"Average IoU: {accuracy_metrics['average_iou']:.3f}")
            print(f"Average MSE: {accuracy_metrics['average_mse']:.3f}")

            # Save final analysis results to file
            save_final_analysis_results(
                accuracy_metrics=accuracy_metrics,
                video_path=self.video_path,
                reference_path=self.reference_path,
                annotations_path=self.annotations_path,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                original_width=self.original_width,
                original_height=self.original_height,
                frame_count=frame_count,
                occlusion_enabled=self.occlusion_handler.occlusion_mask is not None,
                kalman_enabled=self.kalman_manager.use_kalman
            )

        # Show final visualizations
        if show_visualization and last_frame is not None:
            # Draw raw trajectories using visualizer
            raw_trajectory_image = self.visualizer.draw_complete_trajectories(
                last_frame, self.tracker.full_trajectories, use_kalman=False
            )
            if raw_trajectory_image is not None:
                cv2.imshow('Raw Trajectories', raw_trajectory_image)

                # Draw Kalman-filtered trajectories
                if self.kalman_manager.use_kalman:
                    kalman_centers = self.kalman_manager.kalman_centers
                    kalman_trajectory_image = self.visualizer.draw_complete_trajectories(
                        last_frame, self.tracker.full_trajectories, use_kalman=True, kalman_centers=kalman_centers
                    )
                    cv2.imshow('Kalman-filtered Trajectories', kalman_trajectory_image)

                cv2.waitKey(0)  # Wait until a key is pressed

        if show_visualization:
            cv2.destroyAllWindows()

        return accuracy_metrics

    def run_analysis_with_occlusion_tracking(self, with_occlusion_path, without_occlusion_path,
                                           show_visualization=True, save_results=True):
        """
        Run the complete analysis pipeline with occlusion tracking.

        This method replicates the mainFunctionWithOcclusionTracking from final_one.py.

        Parameters:
        -----------
        with_occlusion_path : str
            Path to reference image with occlusion
        without_occlusion_path : str
            Path to reference image without occlusion
        show_visualization : bool
            Whether to show real-time visualization
        save_results : bool
            Whether to save results and visualizations
        """
        print("=== Running Analysis with Occlusion Tracking ===")

        # Compute the occlusion mask using the occlusion handler
        print("Computing occlusion mask...")
        occlusion_mask = self.occlusion_handler.compute_occlusion_mask(
            with_occlusion_path=with_occlusion_path,
            without_occlusion_path=without_occlusion_path
        )
        if occlusion_mask is None:
            print("Failed to compute occlusion mask. Please check the input images.")
            return None

        # Set the occlusion mask in the analyzer
        self.set_occlusion_mask(occlusion_mask)
        print("Occlusion mask computed and set successfully.")

        # Run the analysis
        return self.run_analysis(show_visualization=show_visualization, save_results=save_results)