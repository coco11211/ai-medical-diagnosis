"""
Map visualization for SLAM and coverage planning
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MapperVisualizer:
    """
    Visualizer for SLAM maps, robot trajectory, and coverage planning.
    Windows 11 compatible using matplotlib.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None

        # Color scheme
        self.colors = {
            'free': '#FFFFFF',          # White for free space
            'unknown': '#CCCCCC',       # Gray for unknown
            'occupied': '#000000',      # Black for obstacles
            'robot': '#FF0000',         # Red for robot
            'trajectory': '#0000FF',    # Blue for trajectory
            'path': '#00FF00',          # Green for planned path
            'coverage': '#FFFF00'       # Yellow for coverage area
        }

        logger.info("Visualizer initialized")

    def create_figure(self, num_subplots: int = 1) -> None:
        """Create figure with subplots."""
        if num_subplots == 1:
            self.fig, self.axes = plt.subplots(1, 1, figsize=self.figsize)
            self.axes = [self.axes]
        elif num_subplots == 2:
            self.fig, axes = plt.subplots(1, 2, figsize=self.figsize)
            self.axes = list(axes)
        elif num_subplots == 4:
            self.fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            self.axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
        else:
            self.fig, self.axes = plt.subplots(1, 1, figsize=self.figsize)
            self.axes = [self.axes]

        plt.tight_layout()

    def plot_occupancy_grid(self,
                           occupancy_grid: np.ndarray,
                           ax_idx: int = 0,
                           title: str = "Occupancy Grid Map") -> None:
        """
        Plot occupancy grid.

        Args:
            occupancy_grid: Grid to visualize
            ax_idx: Axis index to plot on
            title: Plot title
        """
        if self.axes is None:
            self.create_figure()

        ax = self.axes[ax_idx]
        ax.clear()

        # Create color map: unknown=gray, free=white, occupied=black
        display_grid = np.zeros_like(occupancy_grid, dtype=float)
        display_grid[occupancy_grid == 50] = 0.5  # Unknown
        display_grid[occupancy_grid < 50] = 1.0   # Free
        display_grid[occupancy_grid >= 100] = 0.0  # Occupied

        ax.imshow(display_grid.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel('X (grid cells)')
        ax.set_ylabel('Y (grid cells)')
        ax.grid(True, alpha=0.3)

    def plot_robot_pose(self,
                       pose: Tuple[float, float, float],
                       ax_idx: int = 0,
                       robot_size: float = 5.0) -> None:
        """
        Plot robot position and orientation.

        Args:
            pose: Robot pose (x, y, theta)
            ax_idx: Axis index
            robot_size: Robot size for visualization
        """
        ax = self.axes[ax_idx]
        x, y, theta = pose

        # Draw robot as circle
        circle = patches.Circle((x, y), robot_size, color=self.colors['robot'],
                               alpha=0.7, zorder=10)
        ax.add_patch(circle)

        # Draw orientation arrow
        arrow_length = robot_size * 1.5
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=robot_size * 0.5,
                head_length=robot_size * 0.5, fc=self.colors['robot'],
                ec=self.colors['robot'], zorder=11)

    def plot_trajectory(self,
                       trajectory: List[np.ndarray],
                       ax_idx: int = 0) -> None:
        """
        Plot robot trajectory.

        Args:
            trajectory: List of poses
            ax_idx: Axis index
        """
        if len(trajectory) < 2:
            return

        ax = self.axes[ax_idx]

        # Extract x, y coordinates
        xs = [pose[0] for pose in trajectory]
        ys = [pose[1] for pose in trajectory]

        ax.plot(xs, ys, color=self.colors['trajectory'], linewidth=2,
               alpha=0.6, label='Trajectory', zorder=5)

    def plot_path(self,
                 path: List[Tuple[int, int]],
                 ax_idx: int = 0,
                 color: Optional[str] = None) -> None:
        """
        Plot planned path.

        Args:
            path: List of waypoints
            ax_idx: Axis index
            color: Path color (optional)
        """
        if len(path) < 2:
            return

        ax = self.axes[ax_idx]
        color = color or self.colors['path']

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        ax.plot(xs, ys, color=color, linewidth=1.5, linestyle='--',
               alpha=0.7, label='Planned Path', zorder=6)

    def plot_sensor_readings(self,
                            pose: Tuple[float, float, float],
                            ranges: List[float],
                            angles: List[float],
                            ax_idx: int = 0,
                            max_range: float = 5.0) -> None:
        """
        Plot sensor beams and detections.

        Args:
            pose: Robot pose
            ranges: Sensor ranges
            angles: Sensor angles
            ax_idx: Axis index
            max_range: Maximum sensor range
        """
        ax = self.axes[ax_idx]
        x, y, theta = pose

        for r, angle in zip(ranges, angles):
            if r >= max_range:
                continue

            abs_angle = theta + angle
            end_x = x + r * np.cos(abs_angle)
            end_y = y + r * np.sin(abs_angle)

            # Draw beam
            ax.plot([x, end_x], [y, end_y], 'r-', alpha=0.2, linewidth=0.5)

            # Draw detection point
            ax.plot(end_x, end_y, 'ro', markersize=2, alpha=0.5)

    def plot_particles(self,
                      particles: np.ndarray,
                      weights: np.ndarray,
                      ax_idx: int = 0) -> None:
        """
        Plot particle filter particles.

        Args:
            particles: Particle positions (N x 3)
            weights: Particle weights (N,)
            ax_idx: Axis index
        """
        ax = self.axes[ax_idx]

        # Scale weights for visualization
        sizes = weights * 1000

        ax.scatter(particles[:, 0], particles[:, 1],
                  s=sizes, c='cyan', alpha=0.5, label='Particles')

    def plot_coverage_area(self,
                          occupancy_grid: np.ndarray,
                          path: List[Tuple[int, int]],
                          robot_width: float,
                          resolution: float,
                          ax_idx: int = 0) -> None:
        """
        Visualize coverage area.

        Args:
            occupancy_grid: Occupancy grid
            path: Coverage path
            robot_width: Robot width
            resolution: Grid resolution
            ax_idx: Axis index
        """
        ax = self.axes[ax_idx]

        # Create coverage map
        coverage_map = np.zeros_like(occupancy_grid)
        radius = int((robot_width / 2) / resolution)

        for x, y in path:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < coverage_map.shape[0] and
                            0 <= ny < coverage_map.shape[1]):
                            coverage_map[nx, ny] = 1

        # Overlay coverage on map
        masked_coverage = np.ma.masked_where(coverage_map == 0, coverage_map)
        ax.imshow(masked_coverage.T, cmap='Greens', alpha=0.3,
                 origin='lower', zorder=7)

    def show_complete_map(self,
                         occupancy_grid: np.ndarray,
                         robot_pose: Tuple[float, float, float],
                         trajectory: List[np.ndarray],
                         path: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Show complete visualization with map, robot, trajectory, and path.

        Args:
            occupancy_grid: Occupancy grid
            robot_pose: Current robot pose
            trajectory: Robot trajectory
            path: Planned path (optional)
        """
        self.create_figure(1)
        self.plot_occupancy_grid(occupancy_grid, 0, "Complete SLAM Map")
        self.plot_trajectory(trajectory, 0)

        if path:
            self.plot_path(path, 0)

        self.plot_robot_pose(robot_pose, 0)

        self.axes[0].legend()
        plt.show()

    def show_comparison(self,
                       occupancy_grid: np.ndarray,
                       robot_pose: Tuple[float, float, float],
                       trajectory: List[np.ndarray],
                       path1: List[Tuple[int, int]],
                       path2: List[Tuple[int, int]],
                       title1: str = "Path 1",
                       title2: str = "Path 2") -> None:
        """
        Compare two different paths side by side.
        """
        self.create_figure(2)

        # First path
        self.plot_occupancy_grid(occupancy_grid, 0, title1)
        self.plot_trajectory(trajectory, 0)
        self.plot_path(path1, 0)
        self.plot_robot_pose(robot_pose, 0)

        # Second path
        self.plot_occupancy_grid(occupancy_grid, 1, title2)
        self.plot_trajectory(trajectory, 1)
        self.plot_path(path2, 1)
        self.plot_robot_pose(robot_pose, 1)

        plt.show()

    def save_figure(self, filename: str, dpi: int = 300) -> None:
        """Save current figure to file."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {filename}")

    def close(self) -> None:
        """Close all figures."""
        plt.close('all')
