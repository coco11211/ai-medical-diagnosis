"""
Object Detection Module
Advanced object detection using deep learning models
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """Detected object information"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    centroid: Tuple[int, int]
    mask: Optional[np.ndarray] = None


class ObjectDetector:
    """
    Object detector using various methods
    """

    def __init__(self, method: str = 'color'):
        """
        Initialize object detector

        Args:
            method: Detection method ('color', 'template', 'feature')
        """
        self.method = method

    def detect_by_color(
        self,
        frame: np.ndarray,
        color_lower: np.ndarray,
        color_upper: np.ndarray,
        min_area: int = 500
    ) -> List[DetectedObject]:
        """
        Detect objects by color range

        Args:
            frame: Input image (BGR)
            color_lower: Lower HSV bound
            color_upper: Upper HSV bound
            min_area: Minimum object area

        Returns:
            List of detected objects
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = cv2.inRange(hsv, color_lower, color_upper)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Centroid
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2

            obj = DetectedObject(
                label='color_object',
                confidence=1.0,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                mask=mask[y:y+h, x:x+w]
            )
            detected_objects.append(obj)

        return detected_objects

    def detect_by_template(
        self,
        frame: np.ndarray,
        template: np.ndarray,
        threshold: float = 0.8
    ) -> List[DetectedObject]:
        """
        Detect objects by template matching

        Args:
            frame: Input image
            template: Template image
            threshold: Matching threshold

        Returns:
            List of detected objects
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Template matching
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Find locations above threshold
        locations = np.where(result >= threshold)

        detected_objects = []
        h, w = template_gray.shape

        for pt in zip(*locations[::-1]):
            x, y = pt
            cx = x + w // 2
            cy = y + h // 2

            # Get confidence from match result
            confidence = result[y, x]

            obj = DetectedObject(
                label='template_match',
                confidence=float(confidence),
                bbox=(x, y, w, h),
                centroid=(cx, cy)
            )
            detected_objects.append(obj)

        # Non-maximum suppression to remove overlapping detections
        detected_objects = self._non_max_suppression(detected_objects, overlap_threshold=0.5)

        return detected_objects

    def detect_edges(
        self,
        frame: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detect edges using Canny edge detector

        Args:
            frame: Input image
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny

        Returns:
            Edge map
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return edges

    def detect_circles(
        self,
        frame: np.ndarray,
        min_radius: int = 10,
        max_radius: int = 100,
        param1: int = 50,
        param2: int = 30
    ) -> List[DetectedObject]:
        """
        Detect circles using Hough Circle Transform

        Args:
            frame: Input image
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            param1: Canny edge threshold
            param2: Accumulator threshold

        Returns:
            List of detected circles
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        detected_objects = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0]:
                cx, cy, r = circle
                x = cx - r
                y = cy - r
                w = h = 2 * r

                obj = DetectedObject(
                    label='circle',
                    confidence=1.0,
                    bbox=(int(x), int(y), int(w), int(h)),
                    centroid=(int(cx), int(cy))
                )
                detected_objects.append(obj)

        return detected_objects

    def detect_aruco_markers(
        self,
        frame: np.ndarray,
        dictionary: int = cv2.aruco.DICT_4X4_50
    ) -> List[Dict]:
        """
        Detect ArUco markers

        Args:
            frame: Input image
            dictionary: ArUco dictionary type

        Returns:
            List of detected markers with IDs and corners
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, _ = detector.detectMarkers(frame)

        markers = []
        if ids is not None:
            for i, corner in enumerate(corners):
                corner_points = corner[0]
                cx = int(np.mean(corner_points[:, 0]))
                cy = int(np.mean(corner_points[:, 1]))

                x = int(np.min(corner_points[:, 0]))
                y = int(np.min(corner_points[:, 1]))
                w = int(np.max(corner_points[:, 0]) - x)
                h = int(np.max(corner_points[:, 1]) - y)

                markers.append({
                    'id': int(ids[i][0]),
                    'corners': corner_points,
                    'centroid': (cx, cy),
                    'bbox': (x, y, w, h)
                })

        return markers

    def _non_max_suppression(
        self,
        objects: List[DetectedObject],
        overlap_threshold: float = 0.5
    ) -> List[DetectedObject]:
        """
        Non-maximum suppression to remove overlapping detections

        Args:
            objects: List of detected objects
            overlap_threshold: Overlap threshold for suppression

        Returns:
            Filtered list of objects
        """
        if len(objects) == 0:
            return []

        # Sort by confidence
        objects = sorted(objects, key=lambda x: x.confidence, reverse=True)

        keep = []

        while len(objects) > 0:
            current = objects[0]
            keep.append(current)
            objects = objects[1:]

            # Remove overlapping objects
            filtered = []
            for obj in objects:
                if self._calculate_iou(current.bbox, obj.bbox) < overlap_threshold:
                    filtered.append(obj)
            objects = filtered

        return keep

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        # IoU
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def draw_detections(
        self,
        frame: np.ndarray,
        objects: List[DetectedObject],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detected objects on frame

        Args:
            frame: Input image
            objects: List of detected objects
            color: Drawing color (BGR)
            thickness: Line thickness

        Returns:
            Annotated image
        """
        result = frame.copy()

        for obj in objects:
            x, y, w, h = obj.bbox
            cx, cy = obj.centroid

            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

            # Draw centroid
            cv2.circle(result, (cx, cy), 5, color, -1)

            # Draw label and confidence
            label = f"{obj.label}: {obj.confidence:.2f}"
            cv2.putText(
                result, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness
            )

        return result
