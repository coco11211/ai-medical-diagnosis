"""
Robot Visualization Module
3D visualization of robot arm using matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Optional, List
import time

from ..kinematics.forward_kinematics import ForwardKinematics
from ..collision.obstacle import Obstacle


class RobotVisualizer:
    """
    3D visualization for robot arm
    """

    def __init__(
        self,
        forward_kinematics: Optional[ForwardKinematics] = None,
        figsize: tuple = (12, 9)
    ):
        """
        Initialize visualizer

        Args:
            forward_kinematics: FK solver
            figsize: Figure size
        """
        self.fk = forward_kinematics or ForwardKinematics()

        # Create figure
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Visualization settings
        self.obstacles: List[Obstacle] = []
        self.workspace_limits = np.array([
            [-1, 1],  # X
            [-1, 1],  # Y
            [0, 1.5]  # Z
        ])

        self._setup_plot()

    def _setup_plot(self):
        """Setup plot appearance"""
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Robot Arm Visualization')

        # Set limits
        self.ax.set_xlim(self.workspace_limits[0])
        self.ax.set_ylim(self.workspace_limits[1])
        self.ax.set_zlim(self.workspace_limits[2])

        # Equal aspect ratio
        self.ax.set_box_aspect([
            self.workspace_limits[0, 1] - self.workspace_limits[0, 0],
            self.workspace_limits[1, 1] - self.workspace_limits[1, 0],
            self.workspace_limits[2, 1] - self.workspace_limits[2, 0]
        ])

        # Grid
        self.ax.grid(True)

    def set_workspace_limits(self, limits: np.ndarray):
        """
        Set workspace visualization limits

        Args:
            limits: Array of shape (3, 2) with [min, max] for each axis
        """
        self.workspace_limits = limits
        self.ax.set_xlim(limits[0])
        self.ax.set_ylim(limits[1])
        self.ax.set_zlim(limits[2])

    def add_obstacle(self, obstacle: Obstacle):
        """Add obstacle to visualization"""
        self.obstacles.append(obstacle)

    def clear_obstacles(self):
        """Clear all obstacles"""
        self.obstacles.clear()

    def plot_robot(
        self,
        joint_angles: np.ndarray,
        show_frame: bool = True,
        show_joints: bool = True,
        link_color: str = 'blue',
        joint_color: str = 'red'
    ):
        """
        Plot robot configuration

        Args:
            joint_angles: Joint angles to visualize
            show_frame: Show coordinate frames
            show_joints: Show joint positions
            link_color: Color for links
            joint_color: Color for joints
        """
        # Clear previous plot
        self.ax.clear()
        self._setup_plot()

        # Get joint positions
        joint_positions = self.fk.get_joint_positions(joint_angles)

        # Plot links
        for i in range(len(joint_positions) - 1):
            p1 = joint_positions[i]
            p2 = joint_positions[i + 1]

            self.ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=link_color,
                linewidth=4,
                marker='o' if show_joints else None,
                markersize=8,
                markerfacecolor=joint_color
            )

        # Plot base
        self.ax.scatter(
            [0], [0], [0],
            color='black',
            s=100,
            marker='s',
            label='Base'
        )

        # Plot end-effector
        end_pos = joint_positions[-1]
        self.ax.scatter(
            [end_pos[0]], [end_pos[1]], [end_pos[2]],
            color='green',
            s=150,
            marker='^',
            label='End-Effector'
        )

        # Plot coordinate frames
        if show_frame:
            transforms = self.fk.compute_all_transforms(joint_angles)
            for T in transforms[::2]:  # Show every other frame for clarity
                self._plot_frame(T, scale=0.1)

        # Plot obstacles
        self._plot_obstacles()

        # Legend
        self.ax.legend()

        plt.draw()

    def plot_trajectory(
        self,
        joint_trajectory: np.ndarray,
        show_path: bool = True,
        show_start_end: bool = True,
        path_color: str = 'orange',
        alpha: float = 0.5
    ):
        """
        Plot trajectory path

        Args:
            joint_trajectory: Array of joint angles (N, num_joints)
            show_path: Show end-effector path
            show_start_end: Highlight start and end positions
            path_color: Color for path
            alpha: Transparency
        """
        if show_path:
            # Get end-effector positions for all trajectory points
            end_effector_path = []

            for angles in joint_trajectory:
                positions = self.fk.get_joint_positions(angles)
                end_effector_path.append(positions[-1])

            end_effector_path = np.array(end_effector_path)

            # Plot path
            self.ax.plot(
                end_effector_path[:, 0],
                end_effector_path[:, 1],
                end_effector_path[:, 2],
                color=path_color,
                linewidth=2,
                alpha=alpha,
                label='Trajectory Path'
            )

            if show_start_end:
                # Start position
                self.ax.scatter(
                    [end_effector_path[0, 0]],
                    [end_effector_path[0, 1]],
                    [end_effector_path[0, 2]],
                    color='green',
                    s=100,
                    marker='o',
                    label='Start'
                )

                # End position
                self.ax.scatter(
                    [end_effector_path[-1, 0]],
                    [end_effector_path[-1, 1]],
                    [end_effector_path[-1, 2]],
                    color='red',
                    s=100,
                    marker='x',
                    label='Goal'
                )

        # Plot current configuration
        self.plot_robot(joint_trajectory[0])

    def animate_trajectory(
        self,
        joint_trajectory: np.ndarray,
        interval: int = 50,
        save_path: Optional[str] = None
    ):
        """
        Animate trajectory execution

        Args:
            joint_trajectory: Array of joint angles
            interval: Time between frames (ms)
            save_path: Path to save animation (optional)
        """
        def update(frame):
            self.plot_robot(joint_trajectory[frame])
            self.ax.set_title(f'Robot Arm Animation - Frame {frame}/{len(joint_trajectory)}')
            return self.ax,

        anim = FuncAnimation(
            self.fig,
            update,
            frames=len(joint_trajectory),
            interval=interval,
            blit=False,
            repeat=True
        )

        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"Animation saved to {save_path}")

        plt.show()

    def _plot_frame(self, transform: np.ndarray, scale: float = 0.1):
        """Plot coordinate frame"""
        origin = transform[:3, 3]
        x_axis = transform[:3, 0] * scale
        y_axis = transform[:3, 1] * scale
        z_axis = transform[:3, 2] * scale

        # X axis - red
        self.ax.quiver(
            origin[0], origin[1], origin[2],
            x_axis[0], x_axis[1], x_axis[2],
            color='red', arrow_length_ratio=0.3, linewidth=1.5
        )

        # Y axis - green
        self.ax.quiver(
            origin[0], origin[1], origin[2],
            y_axis[0], y_axis[1], y_axis[2],
            color='green', arrow_length_ratio=0.3, linewidth=1.5
        )

        # Z axis - blue
        self.ax.quiver(
            origin[0], origin[1], origin[2],
            z_axis[0], z_axis[1], z_axis[2],
            color='blue', arrow_length_ratio=0.3, linewidth=1.5
        )

    def _plot_obstacles(self):
        """Plot all obstacles"""
        for obstacle in self.obstacles:
            if obstacle.obstacle_type.value == 'sphere':
                self._plot_sphere(obstacle)
            elif obstacle.obstacle_type.value == 'box':
                self._plot_box(obstacle)
            elif obstacle.obstacle_type.value == 'cylinder':
                self._plot_cylinder(obstacle)

    def _plot_sphere(self, obstacle: Obstacle):
        """Plot spherical obstacle"""
        center = obstacle.position
        radius = obstacle.dimensions[0]

        # Create sphere mesh
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        self.ax.plot_surface(x, y, z, color='red', alpha=0.3)

    def _plot_box(self, obstacle: Obstacle):
        """Plot box obstacle"""
        center = obstacle.position
        size = obstacle.dimensions

        # Create box vertices
        r = np.array(size) / 2
        vertices = np.array([
            [-r[0], -r[1], -r[2]],
            [r[0], -r[1], -r[2]],
            [r[0], r[1], -r[2]],
            [-r[0], r[1], -r[2]],
            [-r[0], -r[1], r[2]],
            [r[0], -r[1], r[2]],
            [r[0], r[1], r[2]],
            [-r[0], r[1], r[2]]
        ])

        # Transform vertices
        vertices = (obstacle.orientation @ vertices.T).T + center

        # Plot edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Sides
        ]

        for edge in edges:
            points = vertices[edge]
            self.ax.plot(
                points[:, 0], points[:, 1], points[:, 2],
                color='red', linewidth=2, alpha=0.6
            )

    def _plot_cylinder(self, obstacle: Obstacle):
        """Plot cylindrical obstacle"""
        center = obstacle.position
        radius, height = obstacle.dimensions

        # Create cylinder mesh
        theta = np.linspace(0, 2 * np.pi, 20)
        z = np.linspace(-height/2, height/2, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)

        x = center[0] + radius * np.cos(theta_grid)
        y = center[1] + radius * np.sin(theta_grid)
        z = center[2] + z_grid

        self.ax.plot_surface(x, y, z, color='red', alpha=0.3)

    def show(self):
        """Display the plot"""
        plt.show()

    def save_figure(self, filepath: str):
        """Save current figure to file"""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {filepath}")

    def plot_workspace(
        self,
        num_samples: int = 1000,
        alpha: float = 0.1
    ):
        """
        Visualize robot workspace

        Args:
            num_samples: Number of random samples
            alpha: Point transparency
        """
        positions = []

        print(f"Sampling workspace with {num_samples} configurations...")

        for _ in range(num_samples):
            random_angles = np.random.uniform(-np.pi, np.pi, self.fk.num_joints)
            pos, _ = self.fk.compute(random_angles)
            positions.append(pos)

        positions = np.array(positions)

        # Plot workspace points
        self.ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c='blue',
            s=1,
            alpha=alpha,
            label='Workspace'
        )

        self.ax.legend()
        print("Workspace visualization complete")
