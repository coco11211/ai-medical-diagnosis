"""
Lane detection using computer vision
"""
import cv2
import numpy as np


class LaneDetector:
    """Lane detection using computer vision techniques"""

    def __init__(self):
        """Initialize lane detector"""
        self.left_lane = None
        self.right_lane = None
        self.center_lane = None

    def detect_lanes(self, image):
        """
        Detect lanes in the image

        Args:
            image: RGB image from camera

        Returns:
            Dictionary with lane information
        """
        if image is None or image.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Define region of interest (ROI)
        roi_mask = self._create_roi_mask(edges)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=40,
            maxLineGap=100
        )

        # Separate left and right lanes
        left_lines, right_lines = self._separate_lines(lines, image.shape)

        # Fit polynomial to lanes
        left_lane = self._fit_lane(left_lines)
        right_lane = self._fit_lane(right_lines)

        # Calculate center lane
        center_lane = self._calculate_center_lane(left_lane, right_lane)

        self.left_lane = left_lane
        self.right_lane = right_lane
        self.center_lane = center_lane

        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'center_lane': center_lane,
            'lane_width': self._calculate_lane_width(left_lane, right_lane),
            'lane_departure': self._calculate_lane_departure(image.shape, center_lane)
        }

    def _create_roi_mask(self, image):
        """Create region of interest mask"""
        height, width = image.shape
        mask = np.zeros_like(image)

        # Define polygon vertices
        polygon = np.array([[
            (0, height),
            (width // 2 - 50, height // 2),
            (width // 2 + 50, height // 2),
            (width, height)
        ]], dtype=np.int32)

        # Fill polygon
        cv2.fillPoly(mask, polygon, 255)

        return mask

    def _separate_lines(self, lines, image_shape):
        """Separate lines into left and right lanes"""
        if lines is None:
            return [], []

        height, width, _ = image_shape
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope
            if abs(slope) < 0.3:
                continue

            # Classify as left or right
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

        return left_lines, right_lines

    def _fit_lane(self, lines):
        """Fit a polynomial to lane lines"""
        if not lines or len(lines) == 0:
            return None

        # Extract points
        x_points = []
        y_points = []

        for line in lines:
            x1, y1, x2, y2 = line
            x_points.extend([x1, x2])
            y_points.extend([y1, y2])

        if len(x_points) < 2:
            return None

        # Fit polynomial (linear for simplicity)
        try:
            coefficients = np.polyfit(y_points, x_points, 1)
            return coefficients
        except:
            return None

    def _calculate_center_lane(self, left_lane, right_lane):
        """Calculate center lane between left and right"""
        if left_lane is None or right_lane is None:
            return left_lane if left_lane is not None else right_lane

        # Average the coefficients
        center_lane = (left_lane + right_lane) / 2
        return center_lane

    def _calculate_lane_width(self, left_lane, right_lane):
        """Calculate lane width"""
        if left_lane is None or right_lane is None:
            return None

        # Calculate width at a reference y position
        y_ref = 400
        x_left = np.polyval(left_lane, y_ref)
        x_right = np.polyval(right_lane, y_ref)

        return abs(x_right - x_left)

    def _calculate_lane_departure(self, image_shape, center_lane):
        """Calculate lane departure (deviation from center)"""
        if center_lane is None:
            return 0

        height, width, _ = image_shape
        image_center = width // 2

        # Calculate lane center at bottom of image
        y_bottom = height
        lane_center = np.polyval(center_lane, y_bottom)

        # Calculate departure
        departure = lane_center - image_center

        return departure

    def draw_lanes(self, image, lane_data):
        """
        Draw detected lanes on image

        Args:
            image: Input image
            lane_data: Lane detection data

        Returns:
            Image with drawn lanes
        """
        if lane_data is None:
            return image

        output = image.copy()
        height, width = image.shape[:2]

        # Draw left lane
        if lane_data['left_lane'] is not None:
            self._draw_lane_line(output, lane_data['left_lane'], height, (0, 255, 0))

        # Draw right lane
        if lane_data['right_lane'] is not None:
            self._draw_lane_line(output, lane_data['right_lane'], height, (0, 255, 0))

        # Draw center lane
        if lane_data['center_lane'] is not None:
            self._draw_lane_line(output, lane_data['center_lane'], height, (255, 0, 0))

        # Draw lane departure indicator
        departure = lane_data.get('lane_departure', 0)
        cv2.putText(
            output,
            f"Lane Departure: {departure:.1f}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return output

    def _draw_lane_line(self, image, coefficients, height, color):
        """Draw a single lane line"""
        y1 = height
        y2 = int(height * 0.6)

        x1 = int(np.polyval(coefficients, y1))
        x2 = int(np.polyval(coefficients, y2))

        cv2.line(image, (x1, y1), (x2, y2), color, 3)

    def get_steering_suggestion(self, lane_data, image_width):
        """
        Get steering suggestion based on lane detection

        Args:
            lane_data: Lane detection data
            image_width: Width of the image

        Returns:
            Steering angle suggestion in radians
        """
        if lane_data is None or lane_data['center_lane'] is None:
            return 0.0

        departure = lane_data['lane_departure']

        # Simple proportional control
        # Negative departure = too far left, need to steer right
        # Positive departure = too far right, need to steer left
        steering_gain = 0.001
        steering_angle = -departure * steering_gain

        # Limit steering angle
        max_steering = 0.5  # radians
        steering_angle = np.clip(steering_angle, -max_steering, max_steering)

        return steering_angle
