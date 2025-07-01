"""
Object detection module using background subtraction
"""

import cv2
import numpy as np
from ..config.settings import *


class ObjectDetector:
    """Handles object detection using background subtraction"""

    def __init__(self):
        """Initialize the object detector with background subtractor"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY,
            varThreshold=BG_VAR_THRESHOLD,
            detectShadows=BG_DETECT_SHADOWS
        )

    def detect_moving_objects(self, frame):
        """
        Detect moving objects using background subtraction

        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame to process

        Returns:
        --------
        tuple
            (detections, fg_mask) where detections is a list of dicts with bbox, center, area
        """
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
        _, fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and process detected objects
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                # Calculate center point
                center_x = x + w//2
                center_y = y + h//2

                detections.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area
                })

        return detections, fg_mask