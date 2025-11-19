"""
Spiral coverage path planning
Useful for room cleaning starting from center or edges
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class SpiralPlanner:
    """
    Spiral coverage path planner.

    Generates spiral patterns (inward or outward) for area coverage.
    Useful for specific room shapes or starting from known positions.
    """

    def __init__(self, robot_width: float = 0.3, overlap: float = 0.05):
        """
        Initialize spiral planner.

        Args:
            robot_width: Width of the robot (meters)
            overlap: Overlap between spiral loops (meters)
        """
        self.robot_width = robot_width
        self.overlap = overlap
        self.spiral_spacing = robot_width - overlap

        logger.info(f"Spiral planner initialized: robot_width={robot_width}m")

    def plan_outward_spiral(self,
                           occupancy_grid: np.ndarray,
                           start_pose: Tuple[int, int],
                           resolution: float = 0.05) -> List[Tuple[int, int]]:
        """
        Generate outward spiral from start position.

        Args:
            occupancy_grid: Binary occupancy grid
            start_pose: Starting position (x, y)
            resolution: Grid resolution

        Returns:
            List of waypoints
        """
        path = [start_pose]
        spacing = max(1, int(self.spiral_spacing / resolution))

        x, y = start_pose
        dx, dy = spacing, 0  # Start moving right
        steps = 1
        step_count = 0
        turn_count = 0

        visited = set()
        visited.add(start_pose)

        max_iterations = 10000  # Safety limit
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # Move in current direction
            for _ in range(steps):
                x += dx
                y += dy

                # Check bounds
                if not (0 <= x < occupancy_grid.shape[0] and
                       0 <= y < occupancy_grid.shape[1]):
                    return path

                # Check if obstacle
                if occupancy_grid[x, y] >= 100:
                    # Hit obstacle, try to continue around it
                    continue

                if (x, y) not in visited:
                    path.append((x, y))
                    visited.add((x, y))

            # Turn (right for outward spiral)
            dx, dy = -dy, dx
            turn_count += 1

            # Increase step count after two turns
            if turn_count % 2 == 0:
                steps += spacing

            # Check if we've covered enough area
            if len(path) > 100 and self._should_stop_spiral(
                occupancy_grid, x, y, spacing
            ):
                break

        logger.info(f"Generated outward spiral with {len(path)} waypoints")
        return path

    def plan_inward_spiral(self,
                          occupancy_grid: np.ndarray,
                          bounds: Tuple[int, int, int, int],
                          resolution: float = 0.05) -> List[Tuple[int, int]]:
        """
        Generate inward spiral from boundary.

        Args:
            occupancy_grid: Binary occupancy grid
            bounds: Bounding box (min_x, max_x, min_y, max_y)
            resolution: Grid resolution

        Returns:
            List of waypoints
        """
        min_x, max_x, min_y, max_y = bounds
        spacing = max(1, int(self.spiral_spacing / resolution))

        path = []
        visited = set()

        # Start from top-left corner
        x, y = min_x, min_y
        dx, dy = spacing, 0  # Start moving right

        # Define current bounds
        left, right = min_x, max_x
        top, bottom = min_y, max_y

        while left < right and top < bottom:
            # Move right along top
            for x in range(left, right, spacing):
                if occupancy_grid[x, top] < 100 and (x, top) not in visited:
                    path.append((x, top))
                    visited.add((x, top))
            top += spacing

            # Move down along right
            for y in range(top, bottom, spacing):
                if occupancy_grid[right - spacing, y] < 100 and (right - spacing, y) not in visited:
                    path.append((right - spacing, y))
                    visited.add((right - spacing, y))
            right -= spacing

            # Move left along bottom
            if top < bottom:
                for x in range(right, left, -spacing):
                    if occupancy_grid[x, bottom - spacing] < 100 and (x, bottom - spacing) not in visited:
                        path.append((x, bottom - spacing))
                        visited.add((x, bottom - spacing))
                bottom -= spacing

            # Move up along left
            if left < right:
                for y in range(bottom, top, -spacing):
                    if occupancy_grid[left, y] < 100 and (left, y) not in visited:
                        path.append((left, y))
                        visited.add((left, y))
                left += spacing

        logger.info(f"Generated inward spiral with {len(path)} waypoints")
        return path

    def _should_stop_spiral(self,
                           occupancy_grid: np.ndarray,
                           x: int,
                           y: int,
                           radius: int) -> bool:
        """
        Check if spiral should stop (surrounded by obstacles or boundaries).
        """
        # Check if surrounded by obstacles or boundaries
        for dx in [-radius, 0, radius]:
            for dy in [-radius, 0, radius]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # If any neighbor is free and in bounds, continue
                if (0 <= nx < occupancy_grid.shape[0] and
                    0 <= ny < occupancy_grid.shape[1] and
                    occupancy_grid[nx, ny] < 100):
                    return False

        return True

    def plan_room_spiral(self,
                        occupancy_grid: np.ndarray,
                        resolution: float = 0.05) -> List[Tuple[int, int]]:
        """
        Automatically detect room bounds and plan spiral coverage.

        Args:
            occupancy_grid: Occupancy grid
            resolution: Grid resolution

        Returns:
            List of waypoints
        """
        # Find free space bounds
        free_cells = np.argwhere(occupancy_grid < 100)

        if len(free_cells) == 0:
            logger.warning("No free space found")
            return []

        min_x, min_y = free_cells.min(axis=0)
        max_x, max_y = free_cells.max(axis=0)

        # Use inward spiral for room coverage
        return self.plan_inward_spiral(
            occupancy_grid,
            (min_x, max_x, min_y, max_y),
            resolution
        )
