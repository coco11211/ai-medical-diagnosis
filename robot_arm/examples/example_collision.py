"""
Collision avoidance example
Shows how to add obstacles and plan collision-free paths
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from src.control.robot_controller import RobotController
from src.utils.visualizer import RobotVisualizer
from src.collision.obstacle import Obstacle


def main():
    print("Collision Avoidance Example")
    print("-" * 40)

    # Create controller with collision detection
    controller = RobotController(
        enable_vision=False,
        enable_collision_detection=True
    )
    controller.initialize()

    # Create visualizer
    viz = RobotVisualizer(controller.fk)

    # Add obstacles to environment
    print("\nAdding obstacles...")

    obstacle1 = Obstacle.create_sphere(
        center=np.array([0.3, 0.0, 0.3]),
        radius=0.15
    )

    obstacle2 = Obstacle.create_box(
        center=np.array([0.0, 0.4, 0.3]),
        size=(0.15, 0.15, 0.3)
    )

    obstacle3 = Obstacle.create_cylinder(
        center=np.array([-0.2, -0.2, 0.15]),
        radius=0.08,
        height=0.3
    )

    controller.collision_detector.add_obstacle(obstacle1)
    controller.collision_detector.add_obstacle(obstacle2)
    controller.collision_detector.add_obstacle(obstacle3)

    viz.add_obstacle(obstacle1)
    viz.add_obstacle(obstacle2)
    viz.add_obstacle(obstacle3)

    print("Added 3 obstacles:")
    print(f"  - Sphere at {obstacle1.position}")
    print(f"  - Box at {obstacle2.position}")
    print(f"  - Cylinder at {obstacle3.position}")

    # Test collision detection
    print("\n--- Collision Detection Test ---")

    # Safe position
    safe_target = np.array([0.5, 0.0, 0.2])
    print(f"\nTesting safe position: {safe_target}")

    angles, success, error = controller.ik.solve(safe_target)
    if success:
        has_collision, collision_info = controller.collision_detector.check_collision(angles)
        if has_collision:
            print(f"  COLLISION detected: {collision_info}")
        else:
            print("  No collision - safe to move")

            # Move to safe position
            controller.move_to_position(safe_target, check_collision=True)
            print("  Successfully moved to safe position")

    # Collision position
    collision_target = np.array([0.3, 0.0, 0.35])
    print(f"\nTesting collision position: {collision_target}")

    angles, success, error = controller.ik.solve(collision_target)
    if success:
        has_collision, collision_info = controller.collision_detector.check_collision(angles)
        if has_collision:
            print(f"  COLLISION detected: {collision_info}")
            print("  Motion will be blocked")

            # Try to move (should be blocked)
            move_success = controller.move_to_position(collision_target, check_collision=True)
            if not move_success:
                print("  Motion correctly blocked!")
        else:
            print("  No collision detected")

    # Visualize environment
    print("\nVisualizing environment...")
    viz.plot_robot(controller.get_current_joint_angles())
    viz.show()

    controller.shutdown()


if __name__ == "__main__":
    main()
