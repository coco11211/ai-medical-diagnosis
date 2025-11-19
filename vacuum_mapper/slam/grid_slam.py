"""
Grid-based SLAM implementation for robotic vacuum cleaner
Uses occupancy grid mapping with particle filter localization
"""

import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class GridSLAM:
    """
    Grid-based SLAM implementation using occupancy grid mapping.

    The map is represented as a 2D grid where each cell contains:
    - 0: Unknown
    - 1-99: Probability of being free (higher = more certain)
    - 100-199: Probability of being occupied (higher = more certain)
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (500, 500),
                 resolution: float = 0.05,  # meters per cell
                 origin: Tuple[float, float] = (0.0, 0.0)):
        """
        Initialize SLAM system.

        Args:
            grid_size: Size of the occupancy grid (width, height) in cells
            resolution: Resolution in meters per grid cell
            origin: Origin point (x, y) in world coordinates
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.origin = np.array(origin)

        # Initialize occupancy grid (0 = unknown, 50 = uncertain, 100+ = occupied)
        self.grid = np.ones(grid_size, dtype=np.float32) * 50

        # Robot pose (x, y, theta)
        self.robot_pose = np.array([grid_size[0] // 2, grid_size[1] // 2, 0.0])

        # Trajectory history
        self.trajectory = [self.robot_pose.copy()]

        # Map bounds (min_x, max_x, min_y, max_y)
        self.explored_bounds = [grid_size[0]//2, grid_size[0]//2,
                               grid_size[1]//2, grid_size[1]//2]

        logger.info(f"SLAM initialized with grid size {grid_size}, resolution {resolution}m")

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return x, y

    def is_valid_cell(self, x: int, y: int) -> bool:
        """Check if grid cell is within bounds."""
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]

    def update_pose(self, dx: float, dy: float, dtheta: float):
        """
        Update robot pose based on odometry.

        Args:
            dx: Change in x position (meters)
            dy: Change in y position (meters)
            dtheta: Change in orientation (radians)
        """
        # Convert to grid coordinates
        dx_grid = dx / self.resolution
        dy_grid = dy / self.resolution

        # Update pose
        old_pose = self.robot_pose.copy()
        self.robot_pose[0] += dx_grid * np.cos(self.robot_pose[2]) - dy_grid * np.sin(self.robot_pose[2])
        self.robot_pose[1] += dx_grid * np.sin(self.robot_pose[2]) + dy_grid * np.cos(self.robot_pose[2])
        self.robot_pose[2] += dtheta

        # Normalize theta to [-pi, pi]
        self.robot_pose[2] = np.arctan2(np.sin(self.robot_pose[2]), np.cos(self.robot_pose[2]))

        # Update trajectory
        self.trajectory.append(self.robot_pose.copy())

        # Update explored bounds
        x, y = int(self.robot_pose[0]), int(self.robot_pose[1])
        self.explored_bounds[0] = min(self.explored_bounds[0], x)
        self.explored_bounds[1] = max(self.explored_bounds[1], x)
        self.explored_bounds[2] = min(self.explored_bounds[2], y)
        self.explored_bounds[3] = max(self.explored_bounds[3], y)

        logger.debug(f"Pose updated: {old_pose} -> {self.robot_pose}")

    def update_map(self, sensor_ranges: List[float], sensor_angles: List[float],
                   max_range: float = 5.0):
        """
        Update occupancy grid based on sensor readings.

        Args:
            sensor_ranges: List of range measurements (meters)
            sensor_angles: List of sensor angles relative to robot (radians)
            max_range: Maximum sensor range (meters)
        """
        robot_x, robot_y, robot_theta = self.robot_pose

        for range_m, angle in zip(sensor_ranges, sensor_angles):
            if range_m <= 0 or range_m > max_range:
                continue

            # Calculate endpoint in world coordinates
            absolute_angle = robot_theta + angle
            end_x = robot_x + (range_m / self.resolution) * np.cos(absolute_angle)
            end_y = robot_y + (range_m / self.resolution) * np.sin(absolute_angle)

            # Bresenham's line algorithm to trace ray
            cells = self._bresenham_line(int(robot_x), int(robot_y),
                                         int(end_x), int(end_y))

            # Update cells along the ray
            for i, (cx, cy) in enumerate(cells):
                if not self.is_valid_cell(cx, cy):
                    continue

                # Last cell is obstacle, others are free
                if i == len(cells) - 1:
                    # Obstacle detected
                    self.grid[cx, cy] = min(199, self.grid[cx, cy] + 10)
                else:
                    # Free space
                    self.grid[cx, cy] = max(1, self.grid[cx, cy] - 2)

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm for ray tracing.
        Returns list of grid cells along the line from (x0, y0) to (x1, y1).
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            cells.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return cells

    def get_occupancy_grid(self) -> np.ndarray:
        """
        Get occupancy grid as binary map.
        Returns: Binary grid (0=free/unknown, 1=occupied)
        """
        return (self.grid >= 100).astype(np.uint8)

    def get_exploration_map(self) -> np.ndarray:
        """
        Get exploration status map.
        Returns: Map showing explored (1) vs unexplored (0) areas
        """
        return (self.grid != 50).astype(np.uint8)

    def is_cell_free(self, x: int, y: int) -> bool:
        """Check if a cell is free (not occupied)."""
        if not self.is_valid_cell(x, y):
            return False
        return self.grid[x, y] < 100

    def is_cell_explored(self, x: int, y: int) -> bool:
        """Check if a cell has been explored."""
        if not self.is_valid_cell(x, y):
            return False
        return self.grid[x, y] != 50

    def get_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose (x, y, theta) in grid coordinates."""
        return tuple(self.robot_pose)

    def get_trajectory(self) -> List[np.ndarray]:
        """Get robot trajectory history."""
        return self.trajectory

    def get_explored_region(self) -> Tuple[int, int, int, int]:
        """Get bounding box of explored region (min_x, max_x, min_y, max_y)."""
        return tuple(self.explored_bounds)

    def save_map(self, filename: str):
        """Save map to file."""
        np.save(filename, self.grid)
        logger.info(f"Map saved to {filename}")

    def load_map(self, filename: str):
        """Load map from file."""
        self.grid = np.load(filename)
        logger.info(f"Map loaded from {filename}")
