"""
LiDAR sensor simulation for obstacle detection
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class LidarSensor:
    """
    Simulated LiDAR sensor for range measurements.
    Provides 360-degree scanning capability.
    """

    def __init__(self,
                 num_beams: int = 360,
                 max_range: float = 5.0,
                 min_range: float = 0.05,
                 noise_std: float = 0.02,
                 fov: float = 360.0):
        """
        Initialize LiDAR sensor.

        Args:
            num_beams: Number of laser beams
            max_range: Maximum detection range (meters)
            min_range: Minimum detection range (meters)
            noise_std: Standard deviation of measurement noise (meters)
            fov: Field of view in degrees
        """
        self.num_beams = num_beams
        self.max_range = max_range
        self.min_range = min_range
        self.noise_std = noise_std
        self.fov = np.radians(fov)

        # Calculate beam angles
        self.angles = np.linspace(-self.fov / 2, self.fov / 2, num_beams)

        logger.info(f"LiDAR initialized: {num_beams} beams, {max_range}m range")

    def scan(self,
             robot_pose: Tuple[float, float, float],
             occupancy_grid: np.ndarray,
             resolution: float = 0.05) -> Tuple[List[float], List[float]]:
        """
        Perform a LiDAR scan.

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
        """
        Cast a ray to find the distance to the nearest obstacle.

        Args:
            x, y: Starting position in grid coordinates
            theta: Ray direction in radians
            occupancy_grid: Occupancy grid
            resolution: Grid resolution

        Returns:
            Distance to obstacle in meters
        """
        step = 0.5  # Step size in grid cells
        distance_cells = 0.0
        max_cells = self.max_range / resolution

        while distance_cells < max_cells:
            # Current position along ray
            rx = int(x + distance_cells * np.cos(theta))
            ry = int(y + distance_cells * np.sin(theta))

            # Check bounds
            if (rx < 0 or rx >= occupancy_grid.shape[0] or
                ry < 0 or ry >= occupancy_grid.shape[1]):
                return self.max_range

            # Check if obstacle (value >= 100 means occupied)
            if occupancy_grid[rx, ry] >= 100:
                return distance_cells * resolution

            distance_cells += step

        return self.max_range

    def get_beam_angles(self) -> List[float]:
        """Get list of beam angles."""
        return self.angles.tolist()

    def get_max_range(self) -> float:
        """Get maximum sensor range."""
        return self.max_range

    def visualize_scan(self, ranges: List[float], robot_pose: Tuple[float, float, float]) -> List[Tuple[float, float]]:
        """
        Convert scan to list of endpoint coordinates for visualization.

        Args:
            ranges: List of range measurements
            robot_pose: Robot pose (x, y, theta)

        Returns:
            List of (x, y) endpoint coordinates
        """
        x, y, theta = robot_pose
        points = []

        for r, angle in zip(ranges, self.angles):
            if r < self.max_range:
                abs_angle = theta + angle
                px = x + r * np.cos(abs_angle)
                py = y + r * np.sin(abs_angle)
                points.append((px, py))

        return points
