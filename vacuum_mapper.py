"""
Robotic Vacuum Mapper - Main Entry Point
SLAM-based autonomous vacuum cleaner simulation for Windows 11
"""

import argparse
import numpy as np
import time
from typing import Tuple
import logging

from vacuum_mapper.slam import GridSLAM, ParticleFilter
from vacuum_mapper.sensors import LidarSensor, UltrasonicSensor
from vacuum_mapper.coverage import BoustrophedonPlanner, SpiralPlanner
from vacuum_mapper.visualization import MapperVisualizer, RealTimeVisualizer
from vacuum_mapper.utils import Environment, setup_logger


class VacuumMapper:
    """
    Main robotic vacuum mapper controller.
    Integrates SLAM, sensors, coverage planning, and visualization.
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (200, 200),
                 resolution: float = 0.05,
                 robot_width: float = 0.3):
        """
        Initialize vacuum mapper.

        Args:
            grid_size: Map grid size
            resolution: Map resolution (meters/cell)
            robot_width: Robot width in meters
        """
        self.logger = setup_logger('VacuumMapper', logging.INFO)

        # Initialize SLAM
        self.slam = GridSLAM(grid_size, resolution)

        # Initialize sensors
        self.lidar = LidarSensor(num_beams=360, max_range=5.0)
        self.ultrasonic = UltrasonicSensor(num_sensors=8, max_range=0.5)

        # Initialize coverage planners
        self.boustrophedon = BoustrophedonPlanner(robot_width)
        self.spiral = SpiralPlanner(robot_width)

        # Visualization
        self.visualizer = MapperVisualizer()

        # Robot state
        self.robot_width = robot_width
        self.resolution = resolution

        self.logger.info("Vacuum Mapper initialized successfully")

    def explore_and_map(self,
                       environment: Environment,
                       max_steps: int = 1000,
                       visualize: bool = True) -> dict:
        """
        Explore environment and build map using SLAM.

        Args:
            environment: Environment to explore
            max_steps: Maximum exploration steps
            visualize: Whether to visualize in real-time

        Returns:
            Dictionary with mapping results
        """
        self.logger.info("Starting exploration and mapping...")

        # Get environment grid (ground truth)
        env_grid = environment.get_grid()

        # Find starting position
        start_x, start_y = environment.get_random_free_position()
        self.slam.robot_pose = np.array([start_x, start_y, 0.0])

        # Real-time visualization setup
        real_viz = None
        if visualize:
            real_viz = RealTimeVisualizer()
            real_viz.setup(environment.size)
            real_viz.show(block=False)

        # Exploration loop
        for step in range(max_steps):
            # Get current pose
            x, y, theta = self.slam.get_pose()

            # Perform sensor scan
            ranges, angles = self.lidar.scan(
                (x, y, theta),
                env_grid,  # Use ground truth for simulation
                self.resolution
            )

            # Update SLAM map
            self.slam.update_map(ranges, angles, self.lidar.get_max_range())

            # Simple exploration strategy: move forward, turn if obstacle detected
            ultrasonic_ranges, _ = self.ultrasonic.scan((x, y, theta), env_grid, self.resolution)

            if self.ultrasonic.detect_collision(ultrasonic_ranges, threshold=0.15):
                # Obstacle ahead, turn
                dtheta = np.random.uniform(np.pi/4, np.pi/2)
                self.slam.update_pose(0.0, 0.0, dtheta)
            else:
                # Move forward
                dx = 0.1  # 10cm forward
                self.slam.update_pose(dx, 0.0, 0.0)

            # Update visualization
            if visualize and step % 5 == 0:  # Update every 5 steps
                real_viz.update_frame(
                    self.slam.grid,
                    self.slam.robot_pose,
                    self.slam.trajectory,
                    {'ranges': ranges, 'angles': angles, 'max_range': self.lidar.get_max_range()}
                )
                real_viz.pause(0.01)

            # Log progress
            if step % 100 == 0:
                explored = np.sum(self.slam.get_exploration_map())
                total_free = np.sum(env_grid < 100)
                coverage = (explored / total_free * 100) if total_free > 0 else 0
                self.logger.info(f"Step {step}/{max_steps}: {coverage:.1f}% explored")

        if real_viz:
            real_viz.close()

        self.logger.info("Exploration complete")

        return {
            'slam': self.slam,
            'steps': max_steps,
            'explored_percentage': self._calculate_exploration_percentage(env_grid)
        }

    def plan_coverage(self, algorithm: str = 'boustrophedon') -> dict:
        """
        Plan coverage path on mapped area.

        Args:
            algorithm: Coverage algorithm ('boustrophedon' or 'spiral')

        Returns:
            Dictionary with coverage path and metrics
        """
        self.logger.info(f"Planning coverage path using {algorithm}...")

        occupancy_grid = self.slam.get_occupancy_grid()
        start_pose = self.slam.get_pose()
        start_pos = (int(start_pose[0]), int(start_pose[1]))

        if algorithm == 'boustrophedon':
            path = self.boustrophedon.plan(
                self.slam.grid,
                start_pos,
                self.resolution,
                direction='horizontal'
            )
            optimized_path = self.boustrophedon.optimize_path(path)
            coverage_pct = self.boustrophedon.calculate_coverage_percentage(
                self.slam.grid, path, self.resolution
            )
        elif algorithm == 'spiral':
            path = self.spiral.plan_outward_spiral(
                self.slam.grid,
                start_pos,
                self.resolution
            )
            optimized_path = path  # Spiral doesn't need optimization
            coverage_pct = 0.0  # TODO: Implement for spiral
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.logger.info(f"Coverage path generated: {len(path)} waypoints, "
                        f"{len(optimized_path)} after optimization")
        self.logger.info(f"Estimated coverage: {coverage_pct:.1f}%")

        return {
            'path': path,
            'optimized_path': optimized_path,
            'coverage_percentage': coverage_pct,
            'algorithm': algorithm
        }

    def visualize_results(self,
                         show_trajectory: bool = True,
                         show_coverage: bool = True,
                         coverage_path: list = None,
                         save_path: str = None):
        """
        Visualize mapping and coverage results.

        Args:
            show_trajectory: Show robot trajectory
            show_coverage: Show coverage path
            coverage_path: Coverage path to visualize
            save_path: Path to save figure (optional)
        """
        self.visualizer.create_figure(1)
        self.visualizer.plot_occupancy_grid(self.slam.grid, 0, "SLAM Map with Coverage Plan")

        if show_trajectory:
            self.visualizer.plot_trajectory(self.slam.trajectory, 0)

        if show_coverage and coverage_path:
            self.visualizer.plot_path(coverage_path, 0)

        self.visualizer.plot_robot_pose(self.slam.get_pose(), 0)
        self.visualizer.axes[0].legend()

        if save_path:
            self.visualizer.save_figure(save_path)

        self.visualizer.fig.show()
        import matplotlib.pyplot as plt
        plt.show()

    def compare_algorithms(self):
        """Compare boustrophedon and spiral coverage algorithms."""
        self.logger.info("Comparing coverage algorithms...")

        # Plan with both algorithms
        start_pose = self.slam.get_pose()
        start_pos = (int(start_pose[0]), int(start_pose[1]))

        boustro_path = self.boustrophedon.plan(
            self.slam.grid, start_pos, self.resolution, 'horizontal'
        )

        spiral_path = self.spiral.plan_outward_spiral(
            self.slam.grid, start_pos, self.resolution
        )

        # Visualize comparison
        self.visualizer.show_comparison(
            self.slam.grid,
            self.slam.robot_pose,
            self.slam.trajectory,
            boustro_path,
            spiral_path,
            "Boustrophedon Coverage",
            "Spiral Coverage"
        )

    def _calculate_exploration_percentage(self, env_grid: np.ndarray) -> float:
        """Calculate percentage of environment explored."""
        explored = np.sum(self.slam.get_exploration_map())
        total_free = np.sum(env_grid < 100)
        return (explored / total_free * 100) if total_free > 0 else 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Robotic Vacuum Mapper - SLAM and Coverage Planning'
    )

    parser.add_argument(
        '--environment',
        type=str,
        default='furniture',
        choices=['empty', 'furniture', 'l_shape', 'maze', 'multi_room'],
        help='Environment type to simulate'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default='boustrophedon',
        choices=['boustrophedon', 'spiral', 'compare'],
        help='Coverage planning algorithm'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=500,
        help='Number of exploration steps'
    )

    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable real-time visualization during exploration'
    )

    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save result visualization'
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        default=200,
        help='Grid size (creates square grid)'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('main', logging.INFO)
    logger.info("=" * 60)
    logger.info("Robotic Vacuum Mapper - SLAM System")
    logger.info("=" * 60)

    # Create environment
    logger.info(f"Creating {args.environment} environment...")
    env = Environment(size=(args.grid_size, args.grid_size))

    if args.environment == 'empty':
        env.create_empty_room()
    elif args.environment == 'furniture':
        env.create_room_with_furniture(num_obstacles=5)
    elif args.environment == 'l_shape':
        env.create_l_shaped_room()
    elif args.environment == 'maze':
        env.create_maze(wall_density=0.2)
    elif args.environment == 'multi_room':
        env.create_multi_room(num_rooms=(2, 2))

    # Create vacuum mapper
    mapper = VacuumMapper(
        grid_size=(args.grid_size, args.grid_size),
        resolution=0.05,
        robot_width=0.3
    )

    # Explore and map
    result = mapper.explore_and_map(
        env,
        max_steps=args.steps,
        visualize=not args.no_visualize
    )

    logger.info(f"Exploration complete: {result['explored_percentage']:.1f}% mapped")

    # Plan coverage
    if args.algorithm == 'compare':
        mapper.compare_algorithms()
    else:
        coverage_result = mapper.plan_coverage(algorithm=args.algorithm)
        logger.info(f"Coverage planning complete: {coverage_result['coverage_percentage']:.1f}% coverage")

        # Visualize results
        mapper.visualize_results(
            show_trajectory=True,
            show_coverage=True,
            coverage_path=coverage_result['optimized_path'],
            save_path=args.save
        )

    logger.info("=" * 60)
    logger.info("Vacuum Mapper finished successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
