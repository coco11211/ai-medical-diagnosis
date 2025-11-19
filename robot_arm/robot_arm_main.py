"""
Robot Arm Controller - Main Program
Comprehensive robot arm controller with inverse kinematics, trajectory planning,
vision integration, and collision avoidance
"""

import numpy as np
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.control.robot_controller import RobotController
from src.utils.visualizer import RobotVisualizer
from src.trajectory.path_planner import Waypoint
from src.collision.obstacle import Obstacle


def demo_basic_motion():
    """Demonstrate basic robot motion"""
    print("\n" + "="*60)
    print("Demo: Basic Motion Control")
    print("="*60)

    # Create controller
    controller = RobotController(enable_vision=False, enable_collision_detection=False)
    controller.initialize()

    # Create visualizer
    viz = RobotVisualizer(controller.fk)

    # Current position
    print("\nCurrent end-effector position:", controller.get_current_position())

    # Move to home
    print("\nMoving to home position...")
    controller.home_position()

    # Move to target position
    target_pos = np.array([0.4, 0.2, 0.3])
    print(f"\nMoving to target position: {target_pos}")
    success = controller.move_to_position(target_pos)

    if success:
        print("Motion successful!")
        final_pos = controller.get_current_position()
        print(f"Final position: {final_pos}")
        print(f"Position error: {np.linalg.norm(final_pos - target_pos):.6f} m")

        # Visualize final configuration
        viz.plot_robot(controller.get_current_joint_angles())
        viz.show()
    else:
        print("Motion failed!")

    controller.shutdown()


def demo_trajectory_execution():
    """Demonstrate trajectory planning and execution"""
    print("\n" + "="*60)
    print("Demo: Trajectory Execution")
    print("="*60)

    controller = RobotController(enable_vision=False, enable_collision_detection=False)
    controller.initialize()

    viz = RobotVisualizer(controller.fk)

    # Define waypoints
    waypoints = [
        Waypoint(position=np.array([0.3, 0.0, 0.2])),
        Waypoint(position=np.array([0.4, 0.2, 0.3])),
        Waypoint(position=np.array([0.3, 0.3, 0.4])),
        Waypoint(position=np.array([0.2, 0.2, 0.3])),
        Waypoint(position=np.array([0.3, 0.0, 0.2])),
    ]

    print(f"\nExecuting trajectory through {len(waypoints)} waypoints...")

    # Plan and execute
    success = controller.execute_cartesian_path(waypoints, check_collision=False)

    if success:
        print("Trajectory execution successful!")

        # Visualize
        viz.plot_robot(controller.get_current_joint_angles())
        viz.show()
    else:
        print("Trajectory execution failed!")

    controller.shutdown()


def demo_collision_avoidance():
    """Demonstrate collision detection and avoidance"""
    print("\n" + "="*60)
    print("Demo: Collision Avoidance")
    print("="*60)

    controller = RobotController(enable_vision=False, enable_collision_detection=True)
    controller.initialize()

    # Add obstacles
    obstacle1 = Obstacle.create_sphere(center=np.array([0.3, 0.0, 0.3]), radius=0.1)
    obstacle2 = Obstacle.create_box(
        center=np.array([0.0, 0.3, 0.2]),
        size=(0.1, 0.1, 0.3)
    )

    controller.collision_detector.add_obstacle(obstacle1)
    controller.collision_detector.add_obstacle(obstacle2)

    print("\nAdded obstacles:")
    print(f"  - Sphere at {obstacle1.position}, radius {obstacle1.dimensions[0]}")
    print(f"  - Box at {obstacle2.position}, size {obstacle2.dimensions}")

    # Create visualizer and add obstacles
    viz = RobotVisualizer(controller.fk)
    viz.add_obstacle(obstacle1)
    viz.add_obstacle(obstacle2)

    # Try to move to position
    target_pos = np.array([0.35, 0.0, 0.35])
    print(f"\nAttempting to move to: {target_pos}")
    print("This position is inside obstacle 1, should be blocked...")

    success = controller.move_to_position(target_pos, check_collision=True)

    if not success:
        print("Motion correctly blocked due to collision!")

    # Try collision-free position
    safe_target = np.array([0.5, 0.0, 0.2])
    print(f"\nAttempting to move to safe position: {safe_target}")

    success = controller.move_to_position(safe_target, check_collision=True)

    if success:
        print("Motion successful - no collision!")

        # Visualize
        viz.plot_robot(controller.get_current_joint_angles())
        viz.show()

    controller.shutdown()


def demo_pick_and_place():
    """Demonstrate pick and place operation"""
    print("\n" + "="*60)
    print("Demo: Pick and Place Operation")
    print("="*60)

    controller = RobotController(enable_vision=False, enable_collision_detection=False)
    controller.initialize()

    viz = RobotVisualizer(controller.fk)

    # Define pick and place positions
    pick_pos = np.array([0.4, 0.2, 0.1])
    place_pos = np.array([0.4, -0.2, 0.1])

    print(f"\nPick position: {pick_pos}")
    print(f"Place position: {place_pos}")

    # Execute pick and place
    print("\nExecuting pick and place operation...")
    success = controller.pick_and_place(pick_pos, place_pos, approach_height=0.15)

    if success:
        print("Pick and place completed successfully!")

        # Visualize final state
        viz.plot_robot(controller.get_current_joint_angles())
        viz.show()
    else:
        print("Pick and place failed!")

    controller.shutdown()


def demo_workspace_visualization():
    """Visualize robot workspace"""
    print("\n" + "="*60)
    print("Demo: Workspace Visualization")
    print("="*60)

    controller = RobotController()
    controller.initialize()

    viz = RobotVisualizer(controller.fk)

    print("\nCalculating workspace limits...")
    workspace = controller.get_workspace_limits(num_samples=2000)

    print(f"Workspace bounds:")
    print(f"  X: [{workspace['min'][0]:.3f}, {workspace['max'][0]:.3f}]")
    print(f"  Y: [{workspace['min'][1]:.3f}, {workspace['max'][1]:.3f}]")
    print(f"  Z: [{workspace['min'][2]:.3f}, {workspace['max'][2]:.3f}]")
    print(f"  Center: {workspace['center']}")

    # Update visualizer limits
    limits = np.array([
        [workspace['min'][0] - 0.1, workspace['max'][0] + 0.1],
        [workspace['min'][1] - 0.1, workspace['max'][1] + 0.1],
        [workspace['min'][2] - 0.1, workspace['max'][2] + 0.1]
    ])
    viz.set_workspace_limits(limits)

    # Plot workspace
    print("\nGenerating workspace visualization...")
    viz.plot_workspace(num_samples=5000, alpha=0.05)

    # Plot robot in home position
    viz.plot_robot(np.zeros(controller.fk.num_joints))

    viz.show()

    controller.shutdown()


def demo_inverse_kinematics():
    """Demonstrate different IK methods"""
    print("\n" + "="*60)
    print("Demo: Inverse Kinematics Methods")
    print("="*60)

    controller = RobotController()
    controller.initialize()

    target_pos = np.array([0.4, 0.3, 0.5])
    print(f"\nTarget position: {target_pos}")

    methods = ['optimization', 'jacobian', 'ccd']

    for method in methods:
        print(f"\n--- Testing {method.upper()} method ---")

        angles, success, error = controller.ik.solve(
            target_pos,
            initial_guess=np.zeros(controller.fk.num_joints),
            method=method
        )

        if success:
            # Verify solution
            actual_pos, _ = controller.fk.compute(angles)
            print(f"Success! Error: {error:.6f} m")
            print(f"Joint angles (deg): {np.degrees(angles)}")
            print(f"Achieved position: {actual_pos}")
            print(f"Position error: {np.linalg.norm(actual_pos - target_pos):.6f} m")
        else:
            print(f"Failed! Error: {error:.6f} m")

    controller.shutdown()


def interactive_mode():
    """Interactive control mode"""
    print("\n" + "="*60)
    print("Interactive Robot Control Mode")
    print("="*60)

    controller = RobotController(enable_vision=False, enable_collision_detection=True)
    controller.initialize()

    viz = RobotVisualizer(controller.fk)

    print("\nCommands:")
    print("  move <x> <y> <z>  - Move to position")
    print("  home              - Move to home position")
    print("  current           - Show current position")
    print("  joints            - Show current joint angles")
    print("  obstacle sphere <x> <y> <z> <r> - Add sphere obstacle")
    print("  obstacle box <x> <y> <z> <sx> <sy> <sz> - Add box obstacle")
    print("  clear             - Clear obstacles")
    print("  show              - Visualize current state")
    print("  quit              - Exit")

    while True:
        try:
            cmd = input("\n> ").strip().lower().split()

            if not cmd:
                continue

            if cmd[0] == 'quit':
                break

            elif cmd[0] == 'move' and len(cmd) == 4:
                target = np.array([float(cmd[1]), float(cmd[2]), float(cmd[3])])
                print(f"Moving to {target}...")
                success = controller.move_to_position(target, check_collision=True)
                print("Success!" if success else "Failed!")

            elif cmd[0] == 'home':
                controller.home_position()
                print("Moved to home position")

            elif cmd[0] == 'current':
                pos = controller.get_current_position()
                print(f"Current position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

            elif cmd[0] == 'joints':
                angles = controller.get_current_joint_angles()
                print(f"Joint angles (deg): {np.degrees(angles)}")

            elif cmd[0] == 'obstacle':
                if cmd[1] == 'sphere' and len(cmd) == 6:
                    center = np.array([float(cmd[2]), float(cmd[3]), float(cmd[4])])
                    radius = float(cmd[5])
                    obs = Obstacle.create_sphere(center, radius)
                    controller.collision_detector.add_obstacle(obs)
                    viz.add_obstacle(obs)
                    print(f"Added sphere obstacle at {center}, radius {radius}")

                elif cmd[1] == 'box' and len(cmd) == 9:
                    center = np.array([float(cmd[2]), float(cmd[3]), float(cmd[4])])
                    size = (float(cmd[5]), float(cmd[6]), float(cmd[7]))
                    obs = Obstacle.create_box(center, size)
                    controller.collision_detector.add_obstacle(obs)
                    viz.add_obstacle(obs)
                    print(f"Added box obstacle at {center}, size {size}")

            elif cmd[0] == 'clear':
                controller.collision_detector.clear_obstacles()
                viz.clear_obstacles()
                print("Cleared all obstacles")

            elif cmd[0] == 'show':
                viz.plot_robot(controller.get_current_joint_angles())
                viz.show()

            else:
                print("Unknown command")

        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            break

    controller.shutdown()
    print("\nExiting...")


def main():
    parser = argparse.ArgumentParser(description='Robot Arm Controller')
    parser.add_argument(
        'demo',
        nargs='?',
        choices=[
            'basic', 'trajectory', 'collision', 'pick_place',
            'workspace', 'ik', 'interactive'
        ],
        help='Demo to run'
    )

    args = parser.parse_args()

    if args.demo is None:
        print("\n" + "="*60)
        print("Robot Arm Controller")
        print("="*60)
        print("\nAvailable demos:")
        print("  basic        - Basic motion control")
        print("  trajectory   - Trajectory execution")
        print("  collision    - Collision avoidance")
        print("  pick_place   - Pick and place operation")
        print("  workspace    - Workspace visualization")
        print("  ik           - Inverse kinematics methods")
        print("  interactive  - Interactive control mode")
        print("\nUsage: python robot_arm_main.py <demo>")
        return

    demos = {
        'basic': demo_basic_motion,
        'trajectory': demo_trajectory_execution,
        'collision': demo_collision_avoidance,
        'pick_place': demo_pick_and_place,
        'workspace': demo_workspace_visualization,
        'ik': demo_inverse_kinematics,
        'interactive': interactive_mode
    }

    demos[args.demo]()


if __name__ == "__main__":
    main()
