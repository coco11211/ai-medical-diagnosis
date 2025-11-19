"""
Ultrasonic sensor simulation for close-range obstacle detection
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class UltrasonicSensor:
    """
    Simulated ultrasonic sensor array for short-range obstacle detection.
    Typically used for bump detection and close-range navigation.
    """

    def __init__(self,
                 num_sensors: int = 8,
                 max_range: float = 0.5,
                 min_range: float = 0.02,
                 noise_std: float = 0.01,
                 beam_width: float = 30.0):
        """
        Initialize ultrasonic sensor array.

        Args:
            num_sensors: Number of sensors around the robot
            max_range: Maximum detection range (meters)
            min_range: Minimum detection range (meters)
            noise_std: Standard deviation of measurement noise (meters)
            beam_width: Beam width in degrees
        """
        self.num_sensors = num_sensors
        self.max_range = max_range
        self.min_range = min_range
        self.noise_std = noise_std
        self.beam_width = np.radians(beam_width)

        # Sensor positions around robot (evenly distributed)
        self.angles = np.linspace(0, 2 * np.pi, num_sensors, endpoint=False)

        logger.info(f"Ultrasonic array initialized: {num_sensors} sensors, {max_range}m range")

    def scan(self,
             robot_pose: Tuple[float, float, float],
             occupancy_grid: np.ndarray,
             resolution: float = 0.05) -> Tuple[List[float], List[float]]:
        """
        Perform ultrasonic scan.

        Args:
            robot_pose: Robot pose (x, y, theta) in grid coordinates
            occupancy_grid: Occupancy grid
            resolution: Grid resolution (meters per cell)

        Returns:
            Tuple of (ranges, angles) where ranges are in meters
        """
        x, y, theta = robot_pose
        ranges = []

        for angle in self.angles:
            # Absolute angle in world frame
            abs_angle = theta + angle

            # Cast ray to find obstacle
            range_m = self._cast_ray(x, y, abs_angle, occupancy_grid, resolution)

            # Add noise
            if range_m < self.max_range:
                range_m += np.random.randn() * self.noise_std
                range_m = np.clip(range_m, self.min_range, self.max_range)

            ranges.append(range_m)

        return ranges, self.angles.tolist()

    def _cast_ray(self,
                  x: float,
                  y: float,
                  theta: float,
                  occupancy_grid: np.ndarray,
                  resolution: float) -> float:
        """Cast ray to find distance to nearest obstacle."""
        step = 0.2  # Smaller step for short range
        distance_cells = 0.0
        max_cells = self.max_range / resolution

        while distance_cells < max_cells:
            rx = int(x + distance_cells * np.cos(theta))
            ry = int(y + distance_cells * np.sin(theta))

            if (rx < 0 or rx >= occupancy_grid.shape[0] or
                ry < 0 or ry >= occupancy_grid.shape[1]):
                return self.max_range

            if occupancy_grid[rx, ry] >= 100:
                return distance_cells * resolution

            distance_cells += step

        return self.max_range

    def detect_collision(self, ranges: List[float], threshold: float = 0.1) -> bool:
        """
        Check if any sensor detects an obstacle within threshold.

        Args:
            ranges: Sensor range measurements
            threshold: Collision threshold distance (meters)

        Returns:
            True if collision detected
        """
        return any(r < threshold for r in ranges)

    def get_closest_obstacle(self, ranges: List[float]) -> Tuple[float, float]:
        """
        Get distance and angle to closest obstacle.

        Returns:
            Tuple of (distance, angle)
        """
        min_idx = np.argmin(ranges)
        return ranges[min_idx], self.angles[min_idx]
