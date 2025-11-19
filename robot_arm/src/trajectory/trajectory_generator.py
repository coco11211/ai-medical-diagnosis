"""
Trajectory Generator Module
Converts Cartesian space trajectories to joint space trajectories
"""

import numpy as np
from typing import List, Tuple, Optional
from ..kinematics.inverse_kinematics import InverseKinematics
from ..kinematics.forward_kinematics import ForwardKinematics


class TrajectoryGenerator:
    """
    Generates joint space trajectories from Cartesian trajectories
    """

    def __init__(
        self,
        inverse_kinematics: Optional[InverseKinematics] = None,
        forward_kinematics: Optional[ForwardKinematics] = None
    ):
        """
        Initialize trajectory generator

        Args:
            inverse_kinematics: IK solver instance
            forward_kinematics: FK solver instance
        """
        self.fk = forward_kinematics or ForwardKinematics()
        self.ik = inverse_kinematics or InverseKinematics(self.fk)

    def cartesian_to_joint_trajectory(
        self,
        cartesian_positions: np.ndarray,
        initial_joint_angles: Optional[np.ndarray] = None,
        ik_method: str = 'optimization'
    ) -> Tuple[np.ndarray, List[bool], List[float]]:
        """
        Convert Cartesian trajectory to joint trajectory

        Args:
            cartesian_positions: Array of shape (N, 3) with Cartesian positions
            initial_joint_angles: Starting joint configuration
            ik_method: IK solving method

        Returns:
            Tuple of (joint_trajectories, success_flags, errors)
            - joint_trajectories: Array of shape (N, num_joints)
            - success_flags: List of boolean success indicators
            - errors: List of IK errors
        """
        if initial_joint_angles is None:
            initial_joint_angles = np.zeros(self.ik.num_joints)

        num_points = len(cartesian_positions)
        joint_trajectories = np.zeros((num_points, self.ik.num_joints))
        success_flags = []
        errors = []

        current_angles = initial_joint_angles

        for i, target_pos in enumerate(cartesian_positions):
            # Solve IK using previous solution as initial guess
            angles, success, error = self.ik.solve(
                target_pos,
                initial_guess=current_angles,
                method=ik_method
            )

            joint_trajectories[i] = angles
            success_flags.append(success)
            errors.append(error)

            # Update current angles for next iteration
            current_angles = angles

        return joint_trajectories, success_flags, errors

    def generate_joint_trajectory(
        self,
        start_angles: np.ndarray,
        end_angles: np.ndarray,
        num_points: int = 100,
        trajectory_type: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate joint space trajectory between two configurations

        Args:
            start_angles: Starting joint angles
            end_angles: Ending joint angles
            num_points: Number of points in trajectory
            trajectory_type: Type of trajectory ('linear', 'cubic', 'quintic')

        Returns:
            Tuple of (joint_positions, joint_velocities)
        """
        if trajectory_type == 'linear':
            return self._linear_joint_trajectory(start_angles, end_angles, num_points)
        elif trajectory_type == 'cubic':
            return self._cubic_joint_trajectory(start_angles, end_angles, num_points)
        elif trajectory_type == 'quintic':
            return self._quintic_joint_trajectory(start_angles, end_angles, num_points)
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    def _linear_joint_trajectory(
        self,
        start_angles: np.ndarray,
        end_angles: np.ndarray,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linear interpolation in joint space"""
        t = np.linspace(0, 1, num_points)
        positions = np.outer(1 - t, start_angles) + np.outer(t, end_angles)

        # Calculate velocities
        velocities = np.zeros_like(positions)
        dt = 1.0 / (num_points - 1)
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        velocities[0] = (positions[1] - positions[0]) / dt
        velocities[-1] = (positions[-1] - positions[-2]) / dt

        return positions, velocities

    def _cubic_joint_trajectory(
        self,
        start_angles: np.ndarray,
        end_angles: np.ndarray,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cubic polynomial trajectory (zero velocity at endpoints)"""
        t = np.linspace(0, 1, num_points)

        # Cubic polynomial: p(t) = a0 + a1*t + a2*t² + a3*t³
        # Constraints: p(0)=start, p(1)=end, p'(0)=0, p'(1)=0
        a0 = start_angles
        a1 = np.zeros_like(start_angles)
        a2 = 3 * (end_angles - start_angles)
        a3 = -2 * (end_angles - start_angles)

        positions = (a0[:, np.newaxis] +
                    a1[:, np.newaxis] * t +
                    a2[:, np.newaxis] * t**2 +
                    a3[:, np.newaxis] * t**3).T

        # Velocity: v(t) = a1 + 2*a2*t + 3*a3*t²
        velocities = (a1[:, np.newaxis] +
                     2 * a2[:, np.newaxis] * t +
                     3 * a3[:, np.newaxis] * t**2).T

        return positions, velocities

    def _quintic_joint_trajectory(
        self,
        start_angles: np.ndarray,
        end_angles: np.ndarray,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Quintic polynomial trajectory (zero velocity and acceleration at endpoints)"""
        t = np.linspace(0, 1, num_points)

        # Quintic: p(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        # Constraints: p(0)=start, p(1)=end, p'(0)=0, p'(1)=0, p''(0)=0, p''(1)=0
        a0 = start_angles
        a1 = np.zeros_like(start_angles)
        a2 = np.zeros_like(start_angles)
        a3 = 10 * (end_angles - start_angles)
        a4 = -15 * (end_angles - start_angles)
        a5 = 6 * (end_angles - start_angles)

        positions = (a0[:, np.newaxis] +
                    a1[:, np.newaxis] * t +
                    a2[:, np.newaxis] * t**2 +
                    a3[:, np.newaxis] * t**3 +
                    a4[:, np.newaxis] * t**4 +
                    a5[:, np.newaxis] * t**5).T

        # Velocity
        velocities = (a1[:, np.newaxis] +
                     2 * a2[:, np.newaxis] * t +
                     3 * a3[:, np.newaxis] * t**2 +
                     4 * a4[:, np.newaxis] * t**3 +
                     5 * a5[:, np.newaxis] * t**4).T

        return positions, velocities

    def blend_trajectories(
        self,
        traj1: np.ndarray,
        traj2: np.ndarray,
        blend_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Blend two trajectories smoothly

        Args:
            traj1: First trajectory
            traj2: Second trajectory
            blend_ratio: Blending ratio (0 = traj1, 1 = traj2)

        Returns:
            Blended trajectory
        """
        if len(traj1) != len(traj2):
            raise ValueError("Trajectories must have same length")

        return (1 - blend_ratio) * traj1 + blend_ratio * traj2

    def optimize_trajectory_time(
        self,
        joint_trajectory: np.ndarray,
        max_velocities: np.ndarray,
        max_accelerations: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize trajectory timing to respect velocity and acceleration limits

        Args:
            joint_trajectory: Original trajectory
            max_velocities: Maximum velocities for each joint
            max_accelerations: Maximum accelerations for each joint

        Returns:
            Tuple of (time_scaled_trajectory, total_time)
        """
        num_points = len(joint_trajectory)

        # Calculate required velocities between points
        velocities = np.diff(joint_trajectory, axis=0)

        # Find scaling factor based on velocity limits
        max_required_vel = np.max(np.abs(velocities), axis=0)
        vel_scale = np.max(max_required_vel / (max_velocities + 1e-10))

        # Find scaling factor based on acceleration limits
        accelerations = np.diff(velocities, axis=0)
        max_required_acc = np.max(np.abs(accelerations), axis=0)
        acc_scale = np.max(max_required_acc / (max_accelerations + 1e-10))

        # Total time scaling
        time_scale = max(vel_scale, np.sqrt(acc_scale))

        # Generate time stamps
        total_time = (num_points - 1) * time_scale * 0.01  # Assuming 0.01s base time step

        return joint_trajectory, total_time
