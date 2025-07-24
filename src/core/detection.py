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


    def detect_moving_objects_with_shadow_suppression(self, frame):
        """
        Detect moving objects using background subtraction and shadow suppression.

        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame to process

        Returns:
        --------
        tuple
            (detections, fg_mask, non_shadow_mask) where detections is a list of detection dictionaries,
            fg_mask is the foreground mask, and non_shadow_mask is the shadow suppression mask
        """
        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # STEP 1: Shadow suppression — mask low brightness and low saturation
        shadow_v = cv2.inRange(v, 0, 70)             # Low brightness
        shadow_s = cv2.inRange(s, 0, 88)             # Low saturation
        raw_shadow_mask = cv2.bitwise_and(shadow_v, shadow_s)

        # STEP 2: Erode the shadow mask to avoid removing parts of the object
        kernel_erode = np.ones((3, 3), np.uint8)
        eroded_shadow = cv2.erode(raw_shadow_mask, kernel_erode, iterations=1)

        # Invert to get non-shadow areas
        non_shadow_mask = cv2.bitwise_not(eroded_shadow)

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Apply initial morphological operations to remove noise
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Remove shadows from fg_mask using the shadow mask
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=non_shadow_mask)

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

        # Clean blobs from fg_mask before using it
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
        clean_mask = np.zeros_like(fg_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
                clean_mask[labels == i] = 255
        fg_mask = clean_mask

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

        return detections, fg_mask, non_shadow_mask