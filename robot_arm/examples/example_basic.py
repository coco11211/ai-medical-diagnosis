"""
Basic robot control example
Shows how to move the robot to a target position
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from src.control.robot_controller import RobotController
from src.utils.visualizer import RobotVisualizer


def main():
    print("Basic Robot Control Example")
    print("-" * 40)

    # Create controller
    controller = RobotController(
        enable_vision=False,
        enable_collision_detection=False
    )
    controller.initialize()

    # Create visualizer
    viz = RobotVisualizer(controller.fk)

    # Define target position
    target_position = np.array([0.5, 0.3, 0.4])

    print(f"\nTarget position: {target_position}")
    print(f"Current position: {controller.get_current_position()}")

    # Move to target
    print("\nMoving to target...")
    success = controller.move_to_position(target_position)

    if success:
        final_pos = controller.get_current_position()
        error = np.linalg.norm(final_pos - target_position)

        print(f"\nSuccess!")
        print(f"Final position: {final_pos}")
        print(f"Position error: {error*1000:.2f} mm")

        # Visualize
        viz.plot_robot(controller.get_current_joint_angles())
        viz.show()
    else:
        print("\nFailed to reach target!")

    controller.shutdown()


if __name__ == "__main__":
    main()
