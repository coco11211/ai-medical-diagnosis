"""
Boustrophedon (back-and-forth) coverage path planning
Efficient lawn-mower pattern for complete area coverage
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class BoustrophedonPlanner:
    """
    Boustrophedon coverage path planner.

    Generates back-and-forth (lawn mower) patterns to cover all free space.
    Handles obstacles and splits regions into cells for efficient coverage.
    """

    def __init__(self, robot_width: float = 0.3, overlap: float = 0.05):
        """
        Initialize planner.

        Args:
            robot_width: Width of the robot (meters)
            overlap: Overlap between adjacent paths (meters)
        """
        self.robot_width = robot_width
        self.overlap = overlap
        self.path_spacing = robot_width - overlap

        logger.info(f"Boustrophedon planner initialized: robot_width={robot_width}m")

    def plan(self,
             occupancy_grid: np.ndarray,
             start_pose: Tuple[int, int],
             resolution: float = 0.05,
             direction: str = 'horizontal') -> List[Tuple[int, int]]:
        """
        Generate coverage path.

        Args:
            occupancy_grid: Binary occupancy grid (0=free, 1=occupied)
            start_pose: Starting position (x, y) in grid coordinates
            resolution: Grid resolution (meters per cell)
            direction: 'horizontal' or 'vertical' sweep direction

        Returns:
            List of waypoints (x, y) in grid coordinates
        """
        # Create coverage map (0=uncovered, 1=covered, 2=obstacle)
        coverage_map = np.zeros_like(occupancy_grid)
        coverage_map[occupancy_grid >= 100] = 2  # Mark obstacles

        # Calculate stripe width in cells
        stripe_width = max(1, int(self.path_spacing / resolution))

        path = [start_pose]
        current_pos = start_pose

        if direction == 'horizontal':
            path.extend(self._horizontal_coverage(
                coverage_map, current_pos, stripe_width
            ))
        else:
            path.extend(self._vertical_coverage(
                coverage_map, current_pos, stripe_width
            ))

        logger.info(f"Generated path with {len(path)} waypoints")
        return path

    def _horizontal_coverage(self,
                            coverage_map: np.ndarray,
                            start: Tuple[int, int],
                            stripe_width: int) -> List[Tuple[int, int]]:
        """Generate horizontal (left-right) coverage pattern."""
        path = []
        height, width = coverage_map.shape
        y = start[1]
        direction = 1  # 1 = right, -1 = left

        while y < height:
            # Sweep horizontally
            if direction == 1:
                # Left to right
                for x in range(width):
                    if coverage_map[x, y] == 0:  # Free and uncovered
                        path.append((x, y))
                        self._mark_covered(coverage_map, x, y, stripe_width)
            else:
                # Right to left
                for x in range(width - 1, -1, -1):
                    if coverage_map[x, y] == 0:
                        path.append((x, y))
                        self._mark_covered(coverage_map, x, y, stripe_width)

            # Move to next stripe
            y += stripe_width
            direction *= -1  # Alternate direction

            # Add transition waypoint if we have a path
            if path and y < height:
                last_x = path[-1][0]
                path.append((last_x, y))

        return path

    def _vertical_coverage(self,
                          coverage_map: np.ndarray,
                          start: Tuple[int, int],
                          stripe_width: int) -> List[Tuple[int, int]]:
        """Generate vertical (up-down) coverage pattern."""
        path = []
        height, width = coverage_map.shape
        x = start[0]
        direction = 1  # 1 = down, -1 = up

        while x < width:
            # Sweep vertically
            if direction == 1:
                # Top to bottom
                for y in range(height):
                    if coverage_map[x, y] == 0:
                        path.append((x, y))
                        self._mark_covered(coverage_map, x, y, stripe_width)
            else:
                # Bottom to top
                for y in range(height - 1, -1, -1):
                    if coverage_map[x, y] == 0:
                        path.append((x, y))
                        self._mark_covered(coverage_map, x, y, stripe_width)

            # Move to next stripe
            x += stripe_width
            direction *= -1

            if path and x < width:
                last_y = path[-1][1]
                path.append((x, last_y))

        return path

    def _mark_covered(self, coverage_map: np.ndarray, x: int, y: int, width: int):
        """Mark area around point as covered."""
        h, w = coverage_map.shape
        for dx in range(-width // 2, width // 2 + 1):
            for dy in range(-width // 2, width // 2 + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and coverage_map[nx, ny] == 0:
                    coverage_map[nx, ny] = 1

    def optimize_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Optimize path by removing redundant waypoints.

        Args:
            path: Original path

        Returns:
            Optimized path with fewer waypoints
        """
        if len(path) < 3:
            return path

        optimized = [path[0]]

        for i in range(1, len(path) - 1):
            # Check if point is on same line as previous and next
            prev = np.array(path[i - 1])
            curr = np.array(path[i])
            next_pt = np.array(path[i + 1])

            # If not collinear, keep the point
            v1 = curr - prev
            v2 = next_pt - curr

            # Cross product to check collinearity
            cross = v1[0] * v2[1] - v1[1] * v2[0]

            if abs(cross) > 0.1:  # Not collinear
                optimized.append(tuple(curr))

        optimized.append(path[-1])
        return optimized

    def calculate_coverage_percentage(self,
                                     occupancy_grid: np.ndarray,
                                     path: List[Tuple[int, int]],
                                     resolution: float = 0.05) -> float:
        """
        Calculate percentage of free space covered by path.

        Args:
            occupancy_grid: Occupancy grid
            path: Generated path
            resolution: Grid resolution

        Returns:
            Coverage percentage (0-100)
        """
        # Count free cells
        free_cells = np.sum(occupancy_grid < 100)

        if free_cells == 0:
            return 0.0

        # Mark covered cells
        coverage_map = np.zeros_like(occupancy_grid)
        stripe_width = max(1, int(self.path_spacing / resolution))

        for x, y in path:
            self._mark_covered(coverage_map, x, y, stripe_width)

        covered_cells = np.sum(coverage_map == 1)
        percentage = (covered_cells / free_cells) * 100

        return min(100.0, percentage)
