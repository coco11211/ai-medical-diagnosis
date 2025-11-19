"""
Simple demonstration of the Robotic Vacuum Mapper
Shows basic usage of SLAM, mapping, and coverage planning
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vacuum_mapper.slam import GridSLAM
from vacuum_mapper.sensors import LidarSensor
from vacuum_mapper.coverage import BoustrophedonPlanner
from vacuum_mapper.visualization import MapperVisualizer
from vacuum_mapper.utils import Environment, setup_logger


def main():
    """Run simple demo."""
    # Setup logging
    logger = setup_logger('demo', level=20)  # INFO level

    logger.info("=" * 60)
    logger.info("Robotic Vacuum Mapper - Simple Demo")
    logger.info("=" * 60)

    # Create environment
    logger.info("Creating test environment...")
    env = Environment(size=(150, 150))
    env.create_room_with_furniture(num_obstacles=3)

    # Initialize SLAM
    logger.info("Initializing SLAM system...")
    slam = GridSLAM(grid_size=(150, 150), resolution=0.05)

    # Initialize sensor
    lidar = LidarSensor(num_beams=180, max_range=5.0)

    # Set starting position
    start_x, start_y = 75, 75
    slam.robot_pose[0] = start_x
    slam.robot_pose[1] = start_y

    # Simulate some exploration
    logger.info("Simulating exploration (50 steps)...")
    import numpy as np

    for step in range(50):
        # Get pose
        x, y, theta = slam.get_pose()

        # Scan
        ranges, angles = lidar.scan((x, y, theta), env.get_grid(), 0.05)

        # Update map
        slam.update_map(ranges, angles, lidar.get_max_range())

        # Simple movement: move forward or turn
        if step % 10 == 0:
            # Turn
            slam.update_pose(0, 0, np.pi/4)
        else:
            # Move forward
            slam.update_pose(0.1, 0, 0)

    logger.info(f"Exploration complete. Mapped {np.sum(slam.get_exploration_map())} cells")

    # Plan coverage
    logger.info("Planning coverage path...")
    planner = BoustrophedonPlanner(robot_width=0.3)
    path = planner.plan(slam.grid, (int(start_x), int(start_y)), 0.05)
    logger.info(f"Generated path with {len(path)} waypoints")

    # Visualize
    logger.info("Generating visualization...")
    viz = MapperVisualizer()
    viz.create_figure(1)
    viz.plot_occupancy_grid(slam.grid, 0, "SLAM Map with Coverage Path")
    viz.plot_trajectory(slam.trajectory, 0)
    viz.plot_path(path, 0)
    viz.plot_robot_pose(slam.get_pose(), 0)
    viz.axes[0].legend()

    logger.info("Demo complete! Close the window to exit.")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    main()
