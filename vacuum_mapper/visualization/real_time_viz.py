"""
Real-time visualization for live mapping and navigation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class RealTimeVisualizer:
    """
    Real-time visualizer for live SLAM and navigation.
    Updates visualization as robot explores environment.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 8), update_interval: int = 100):
        """
        Initialize real-time visualizer.

        Args:
            figsize: Figure size
            update_interval: Update interval in milliseconds
        """
        self.figsize = figsize
        self.update_interval = update_interval

        self.fig = None
        self.ax = None
        self.animation = None

        # Plot elements
        self.map_img = None
        self.robot_marker = None
        self.trajectory_line = None
        self.sensor_lines = []

        logger.info("Real-time visualizer initialized")

    def setup(self, grid_size: Tuple[int, int]) -> None:
        """
        Setup visualization window.

        Args:
            grid_size: Size of the grid to visualize
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(0, grid_size[0])
        self.ax.set_ylim(0, grid_size[1])
        self.ax.set_aspect('equal')
        self.ax.set_title('Real-Time SLAM Mapping')
        self.ax.set_xlabel('X (cells)')
        self.ax.set_ylabel('Y (cells)')
        self.ax.grid(True, alpha=0.3)

        # Initialize empty map
        empty_grid = np.ones(grid_size) * 0.5
        self.map_img = self.ax.imshow(empty_grid.T, cmap='gray',
                                      origin='lower', vmin=0, vmax=1,
                                      alpha=0.7)

        # Initialize robot marker
        self.robot_marker, = self.ax.plot([], [], 'ro', markersize=10,
                                          label='Robot', zorder=10)

        # Initialize trajectory line
        self.trajectory_line, = self.ax.plot([], [], 'b-', linewidth=2,
                                             alpha=0.6, label='Trajectory')

        self.ax.legend()
        plt.tight_layout()

    def update_frame(self,
                    occupancy_grid: np.ndarray,
                    robot_pose: Tuple[float, float, float],
                    trajectory: List[np.ndarray],
                    sensor_data: Optional[dict] = None) -> None:
        """
        Update visualization frame.

        Args:
            occupancy_grid: Current occupancy grid
            robot_pose: Current robot pose
            trajectory: Robot trajectory history
            sensor_data: Optional sensor visualization data
        """
        # Update occupancy grid
        display_grid = np.zeros_like(occupancy_grid, dtype=float)
        display_grid[occupancy_grid == 50] = 0.5  # Unknown
        display_grid[occupancy_grid < 50] = 1.0   # Free
        display_grid[occupancy_grid >= 100] = 0.0  # Occupied

        self.map_img.set_data(display_grid.T)

        # Update robot position
        x, y, theta = robot_pose
        self.robot_marker.set_data([x], [y])

        # Update trajectory
        if len(trajectory) > 1:
            xs = [pose[0] for pose in trajectory]
            ys = [pose[1] for pose in trajectory]
            self.trajectory_line.set_data(xs, ys)

        # Update sensor visualization if provided
        if sensor_data:
            self._update_sensors(robot_pose, sensor_data)

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_sensors(self, robot_pose: Tuple[float, float, float], sensor_data: dict):
        """Update sensor beam visualization."""
        # Clear old sensor lines
        for line in self.sensor_lines:
            line.remove()
        self.sensor_lines.clear()

        x, y, theta = robot_pose
        ranges = sensor_data.get('ranges', [])
        angles = sensor_data.get('angles', [])
        max_range = sensor_data.get('max_range', 5.0)

        for r, angle in zip(ranges, angles):
            if r >= max_range:
                continue

            abs_angle = theta + angle
            end_x = x + r * np.cos(abs_angle)
            end_y = y + r * np.sin(abs_angle)

            line, = self.ax.plot([x, end_x], [y, end_y], 'r-',
                                alpha=0.3, linewidth=0.5)
            self.sensor_lines.append(line)

    def start_animation(self, update_func: Callable, interval: Optional[int] = None) -> None:
        """
        Start animation loop.

        Args:
            update_func: Function to call for each frame (should return updated data)
            interval: Update interval in milliseconds (uses default if None)
        """
        if interval is None:
            interval = self.update_interval

        def animate(frame):
            data = update_func()
            if data:
                self.update_frame(**data)

        self.animation = FuncAnimation(self.fig, animate, interval=interval,
                                      blit=False, cache_frame_data=False)

    def show(self, block: bool = True) -> None:
        """
        Show visualization window.

        Args:
            block: Whether to block execution
        """
        plt.ion() if not block else plt.ioff()
        plt.show(block=block)

    def pause(self, duration: float) -> None:
        """
        Pause for specified duration.

        Args:
            duration: Pause duration in seconds
        """
        plt.pause(duration)

    def close(self) -> None:
        """Close visualization window."""
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)
