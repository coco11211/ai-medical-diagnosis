"""
Environment simulation for testing vacuum mapper
"""

import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class Environment:
    """
    Simulated environment for robotic vacuum testing.
    Provides various room layouts with obstacles.
    """

    def __init__(self, size: Tuple[int, int] = (200, 200), resolution: float = 0.05):
        """
        Initialize environment.

        Args:
            size: Environment size in grid cells (width, height)
            resolution: Resolution in meters per cell
        """
        self.size = size
        self.resolution = resolution
        self.grid = np.zeros(size, dtype=np.float32)

        logger.info(f"Environment created: {size} cells, {resolution}m resolution")

    def create_empty_room(self, wall_thickness: int = 5) -> np.ndarray:
        """
        Create empty rectangular room with walls.

        Args:
            wall_thickness: Thickness of walls in cells

        Returns:
            Occupancy grid with walls
        """
        self.grid = np.ones(self.size) * 50  # Unknown initially

        # Create walls
        self.grid[:wall_thickness, :] = 150  # Left wall
        self.grid[-wall_thickness:, :] = 150  # Right wall
        self.grid[:, :wall_thickness] = 150  # Bottom wall
        self.grid[:, -wall_thickness:] = 150  # Top wall

        logger.info("Created empty room environment")
        return self.grid.copy()

    def create_room_with_furniture(self, num_obstacles: int = 5) -> np.ndarray:
        """
        Create room with random furniture obstacles.

        Args:
            num_obstacles: Number of obstacles to place

        Returns:
            Occupancy grid with obstacles
        """
        self.create_empty_room()

        # Add random rectangular obstacles (furniture)
        for _ in range(num_obstacles):
            # Random obstacle size
            width = np.random.randint(10, 30)
            height = np.random.randint(10, 30)

            # Random position (not too close to walls)
            x = np.random.randint(20, self.size[0] - width - 20)
            y = np.random.randint(20, self.size[1] - height - 20)

            # Place obstacle
            self.grid[x:x+width, y:y+height] = 150

        logger.info(f"Created room with {num_obstacles} obstacles")
        return self.grid.copy()

    def create_l_shaped_room(self) -> np.ndarray:
        """
        Create L-shaped room layout.

        Returns:
            Occupancy grid for L-shaped room
        """
        self.grid = np.ones(self.size) * 150  # Start with all walls

        # Create L-shape (two rectangles)
        # Horizontal part
        self.grid[10:self.size[0]-10, 10:self.size[1]//2] = 50

        # Vertical part
        self.grid[10:self.size[0]//2, self.size[1]//2:self.size[1]-10] = 50

        logger.info("Created L-shaped room environment")
        return self.grid.copy()

    def create_maze(self, wall_density: float = 0.3) -> np.ndarray:
        """
        Create maze-like environment.

        Args:
            wall_density: Density of walls (0-1)

        Returns:
            Occupancy grid with maze
        """
        self.create_empty_room()

        # Add random walls
        for _ in range(int(wall_density * 100)):
            # Random wall orientation
            if np.random.rand() < 0.5:
                # Horizontal wall
                length = np.random.randint(20, 50)
                x = np.random.randint(15, self.size[0] - length - 15)
                y = np.random.randint(15, self.size[1] - 15)
                self.grid[x:x+length, y:y+3] = 150
            else:
                # Vertical wall
                length = np.random.randint(20, 50)
                x = np.random.randint(15, self.size[0] - 15)
                y = np.random.randint(15, self.size[1] - length - 15)
                self.grid[x:x+3, y:y+length] = 150

        logger.info("Created maze environment")
        return self.grid.copy()

    def create_multi_room(self, num_rooms: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        Create multi-room layout with doorways.

        Args:
            num_rooms: Number of rooms (rows, cols)

        Returns:
            Occupancy grid with multiple rooms
        """
        self.create_empty_room()

        rows, cols = num_rooms
        room_width = (self.size[0] - 20) // cols
        room_height = (self.size[1] - 20) // rows

        # Create room dividers
        for i in range(1, rows):
            y = 10 + i * room_height
            # Horizontal wall with doorways
            self.grid[:, y:y+3] = 150

            # Add doorways
            for j in range(cols):
                door_x = 10 + j * room_width + room_width // 2
                self.grid[door_x-5:door_x+5, y:y+3] = 50

        for i in range(1, cols):
            x = 10 + i * room_width
            # Vertical wall with doorways
            self.grid[x:x+3, :] = 150

            # Add doorways
            for j in range(rows):
                door_y = 10 + j * room_height + room_height // 2
                self.grid[x:x+3, door_y-5:door_y+5] = 50

        logger.info(f"Created multi-room environment ({rows}x{cols})")
        return self.grid.copy()

    def add_circular_obstacle(self, center: Tuple[int, int], radius: int):
        """
        Add circular obstacle to environment.

        Args:
            center: Center position (x, y)
            radius: Obstacle radius
        """
        x_c, y_c = center

        for x in range(max(0, x_c - radius), min(self.size[0], x_c + radius + 1)):
            for y in range(max(0, y_c - radius), min(self.size[1], y_c + radius + 1)):
                if (x - x_c)**2 + (y - y_c)**2 <= radius**2:
                    self.grid[x, y] = 150

    def get_random_free_position(self) -> Tuple[int, int]:
        """
        Get random free position in environment.

        Returns:
            Random (x, y) position in free space
        """
        free_cells = np.argwhere(self.grid < 100)

        if len(free_cells) == 0:
            # Default to center if no free cells
            return self.size[0] // 2, self.size[1] // 2

        idx = np.random.randint(0, len(free_cells))
        return tuple(free_cells[idx])

    def is_collision(self, x: int, y: int, robot_radius: int = 5) -> bool:
        """
        Check if position would cause collision.

        Args:
            x, y: Position to check
            robot_radius: Robot radius in cells

        Returns:
            True if collision detected
        """
        for dx in range(-robot_radius, robot_radius + 1):
            for dy in range(-robot_radius, robot_radius + 1):
                if dx*dx + dy*dy > robot_radius*robot_radius:
                    continue

                nx, ny = x + dx, y + dy

                if (nx < 0 or nx >= self.size[0] or
                    ny < 0 or ny >= self.size[1] or
                    self.grid[nx, ny] >= 100):
                    return True

        return False

    def get_grid(self) -> np.ndarray:
        """Get current environment grid."""
        return self.grid.copy()
