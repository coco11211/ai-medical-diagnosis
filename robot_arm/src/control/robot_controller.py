"""
Robot Controller Module
Main controller integrating all subsystems
"""

import numpy as np
import time
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from ..kinematics.forward_kinematics import ForwardKinematics
from ..kinematics.inverse_kinematics import InverseKinematics
from ..trajectory.path_planner import PathPlanner, Waypoint, InterpolationType
from ..trajectory.trajectory_generator import TrajectoryGenerator
from ..vision.vision_system import VisionSystem
from ..collision.collision_detector import CollisionDetector
from ..collision.obstacle import Obstacle


@dataclass
class RobotState:
    """Current robot state"""
    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    end_effector_position: np.ndarray
    end_effector_orientation: np.ndarray
    timestamp: float


class RobotController:
    """
    Main robot arm controller integrating all subsystems
    """

    def __init__(
        self,
        enable_vision: bool = False,
        enable_collision_detection: bool = True,
        control_frequency: float = 100.0  # Hz
    ):
        """
        Initialize robot controller

        Args:
            enable_vision: Enable vision system
            enable_collision_detection: Enable collision detection
            control_frequency: Control loop frequency (Hz)
        """
        # Initialize subsystems
        self.fk = ForwardKinematics()
        self.ik = InverseKinematics(self.fk)
        self.path_planner = PathPlanner()
        self.trajectory_generator = TrajectoryGenerator(self.ik, self.fk)
        self.collision_detector = CollisionDetector(self.fk) if enable_collision_detection else None
        self.vision_system = VisionSystem() if enable_vision else None

        # Robot state
        self.current_state = RobotState(
            joint_angles=np.zeros(self.fk.num_joints),
            joint_velocities=np.zeros(self.fk.num_joints),
            end_effector_position=np.zeros(3),
            end_effector_orientation=np.eye(3),
            timestamp=time.time()
        )

        # Control parameters
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency

        # Safety limits
        self.max_joint_velocities = np.array([np.pi] * self.fk.num_joints)  # rad/s
        self.max_joint_accelerations = np.array([2*np.pi] * self.fk.num_joints)  # rad/s²

        # Flags
        self.is_executing = False
        self.emergency_stop = False

        print("Robot Controller initialized")
        print(f"- Degrees of Freedom: {self.fk.num_joints}")
        print(f"- Vision System: {'Enabled' if enable_vision else 'Disabled'}")
        print(f"- Collision Detection: {'Enabled' if enable_collision_detection else 'Disabled'}")
        print(f"- Control Frequency: {control_frequency} Hz")

    def initialize(self) -> bool:
        """
        Initialize all subsystems

        Returns:
            True if successful
        """
        success = True

        if self.vision_system:
            if not self.vision_system.initialize():
                print("Warning: Vision system initialization failed")
                success = False

        # Update initial state
        self.update_state()

        return success

    def update_state(self):
        """Update current robot state"""
        pos, orient = self.fk.compute(self.current_state.joint_angles)

        self.current_state.end_effector_position = pos
        self.current_state.end_effector_orientation = orient
        self.current_state.timestamp = time.time()

    def get_current_position(self) -> np.ndarray:
        """Get current end-effector position"""
        return self.current_state.end_effector_position.copy()

    def get_current_joint_angles(self) -> np.ndarray:
        """Get current joint angles"""
        return self.current_state.joint_angles.copy()

    def move_to_position(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        check_collision: bool = True,
        ik_method: str = 'optimization'
    ) -> bool:
        """
        Move end-effector to target position

        Args:
            target_position: Target [x, y, z]
            target_orientation: Target orientation (optional)
            check_collision: Check for collisions
            ik_method: IK solving method

        Returns:
            True if successful
        """
        if self.emergency_stop:
            print("Emergency stop active")
            return False

        # Solve inverse kinematics
        target_angles, success, error = self.ik.solve(
            target_position,
            target_orientation,
            initial_guess=self.current_state.joint_angles,
            method=ik_method
        )

        if not success:
            print(f"IK failed: error = {error:.4f}")
            return False

        # Check collision if enabled
        if check_collision and self.collision_detector:
            has_collision, collision_info = self.collision_detector.check_collision(target_angles)
            if has_collision:
                print(f"Target configuration in collision: {collision_info}")
                return False

        # Execute motion
        return self.move_to_joint_angles(target_angles, check_collision)

    def move_to_joint_angles(
        self,
        target_angles: np.ndarray,
        check_collision: bool = True
    ) -> bool:
        """
        Move to target joint configuration

        Args:
            target_angles: Target joint angles
            check_collision: Check for collisions

        Returns:
            True if successful
        """
        if self.emergency_stop:
            print("Emergency stop active")
            return False

        # Generate trajectory
        joint_traj, joint_vel = self.trajectory_generator.generate_joint_trajectory(
            self.current_state.joint_angles,
            target_angles,
            num_points=100,
            trajectory_type='quintic'
        )

        # Check trajectory collision
        if check_collision and self.collision_detector:
            has_collision, collision_indices = self.collision_detector.check_trajectory_collision(
                joint_traj, check_interval=5
            )
            if has_collision:
                print(f"Trajectory has collision at indices: {collision_indices}")
                return False

        # Execute trajectory
        return self.execute_joint_trajectory(joint_traj, joint_vel)

    def execute_joint_trajectory(
        self,
        joint_trajectory: np.ndarray,
        joint_velocities: Optional[np.ndarray] = None
    ) -> bool:
        """
        Execute a joint space trajectory

        Args:
            joint_trajectory: Array of joint angles (N, num_joints)
            joint_velocities: Array of joint velocities (optional)

        Returns:
            True if successful
        """
        if self.is_executing:
            print("Already executing trajectory")
            return False

        self.is_executing = True

        try:
            for i, target_angles in enumerate(joint_trajectory):
                if self.emergency_stop:
                    print("Emergency stop triggered")
                    self.is_executing = False
                    return False

                # Update joint angles
                self.current_state.joint_angles = target_angles

                # Update velocities if provided
                if joint_velocities is not None:
                    self.current_state.joint_velocities = joint_velocities[i]

                # Update state
                self.update_state()

                # Simulate control loop timing
                time.sleep(self.dt)

                # Print progress every 20 steps
                if i % 20 == 0:
                    pos = self.current_state.end_effector_position
                    print(f"Progress: {i}/{len(joint_trajectory)} - Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

            print("Trajectory execution completed")
            self.is_executing = False
            return True

        except Exception as e:
            print(f"Error executing trajectory: {e}")
            self.is_executing = False
            return False

    def execute_cartesian_path(
        self,
        waypoints: List[Waypoint],
        interpolation_type: InterpolationType = InterpolationType.CUBIC_SPLINE,
        check_collision: bool = True
    ) -> bool:
        """
        Execute Cartesian space path through waypoints

        Args:
            waypoints: List of waypoints to visit
            interpolation_type: Trajectory interpolation method
            check_collision: Check for collisions

        Returns:
            True if successful
        """
        # Plan path
        self.path_planner.interpolation_type = interpolation_type
        positions, velocities, times = self.path_planner.plan_path(waypoints)

        print(f"Planned path with {len(positions)} points")

        # Convert to joint trajectory
        joint_traj, success_flags, errors = self.trajectory_generator.cartesian_to_joint_trajectory(
            positions,
            initial_joint_angles=self.current_state.joint_angles
        )

        # Check IK success
        failed_points = [i for i, success in enumerate(success_flags) if not success]
        if failed_points:
            print(f"IK failed at {len(failed_points)} points: {failed_points[:5]}...")
            return False

        # Check collisions
        if check_collision and self.collision_detector:
            has_collision, collision_indices = self.collision_detector.check_trajectory_collision(
                joint_traj, check_interval=5
            )
            if has_collision:
                print(f"Path has collision at {len(collision_indices)} points")
                return False

        # Execute
        return self.execute_joint_trajectory(joint_traj)

    def pick_and_place(
        self,
        pick_position: np.ndarray,
        place_position: np.ndarray,
        approach_height: float = 0.1,
        check_collision: bool = True
    ) -> bool:
        """
        Execute pick and place operation

        Args:
            pick_position: Object position to pick
            place_position: Target position to place
            approach_height: Height above object for approach
            check_collision: Check for collisions

        Returns:
            True if successful
        """
        print("Executing pick and place operation...")

        # Create waypoints
        waypoints = [
            # Current position
            Waypoint(position=self.get_current_position()),

            # Approach pick position
            Waypoint(position=pick_position + np.array([0, 0, approach_height])),

            # Pick position
            Waypoint(position=pick_position),

            # Lift after pick
            Waypoint(position=pick_position + np.array([0, 0, approach_height])),

            # Approach place position
            Waypoint(position=place_position + np.array([0, 0, approach_height])),

            # Place position
            Waypoint(position=place_position),

            # Retreat after place
            Waypoint(position=place_position + np.array([0, 0, approach_height]))
        ]

        # Execute path
        return self.execute_cartesian_path(waypoints, check_collision=check_collision)

    def vision_guided_grasp(
        self,
        detect_method: str = 'color',
        **detect_kwargs
    ) -> bool:
        """
        Perform vision-guided grasping

        Args:
            detect_method: Object detection method
            **detect_kwargs: Detection parameters

        Returns:
            True if successful
        """
        if not self.vision_system or not self.vision_system.is_initialized:
            print("Vision system not available")
            return False

        # Capture image
        frame = self.vision_system.capture_frame()
        if frame is None:
            print("Failed to capture image")
            return False

        # Detect objects
        objects = self.vision_system.detect_objects(frame, method=detect_method)

        if not objects:
            print("No objects detected")
            return False

        # Select first object
        obj = objects[0]
        pixel_coords = obj['centroid']

        print(f"Detected object at pixel: {pixel_coords}")

        # Convert to world coordinates (requires depth and calibration)
        # For now, use placeholder depth
        depth = 0.5  # meters
        camera_coords = self.vision_system.pixel_to_camera_coordinates(
            np.array(pixel_coords), depth
        )

        # Transform to world coordinates (if calibrated)
        if self.vision_system.calibration and self.vision_system.calibration.rvec is not None:
            target_position = self.vision_system.camera_to_world_coordinates(camera_coords)
        else:
            print("Camera not calibrated, using camera coordinates")
            target_position = camera_coords

        print(f"Target position: {target_position}")

        # Move to grasp position
        return self.move_to_position(target_position)

    def home_position(self) -> bool:
        """Move to home position (all joints at zero)"""
        print("Moving to home position...")
        return self.move_to_joint_angles(np.zeros(self.fk.num_joints))

    def emergency_stop_trigger(self):
        """Trigger emergency stop"""
        print("EMERGENCY STOP TRIGGERED")
        self.emergency_stop = True
        self.is_executing = False

    def reset_emergency_stop(self):
        """Reset emergency stop"""
        print("Emergency stop reset")
        self.emergency_stop = False

    def get_workspace_limits(
        self,
        num_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Estimate workspace limits by sampling random configurations

        Args:
            num_samples: Number of random samples

        Returns:
            Dictionary with min/max positions
        """
        positions = []

        for _ in range(num_samples):
            random_angles = np.random.uniform(-np.pi, np.pi, self.fk.num_joints)
            pos, _ = self.fk.compute(random_angles)
            positions.append(pos)

        positions = np.array(positions)

        return {
            'min': np.min(positions, axis=0),
            'max': np.max(positions, axis=0),
            'center': np.mean(positions, axis=0)
        }

    def shutdown(self):
        """Shutdown controller and release resources"""
        print("Shutting down robot controller...")

        if self.vision_system:
            self.vision_system.release()

        self.is_executing = False
        print("Shutdown complete")
