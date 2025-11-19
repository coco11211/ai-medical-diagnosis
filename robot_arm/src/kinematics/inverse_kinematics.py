"""
Inverse Kinematics Module
Computes joint angles to achieve desired end-effector position and orientation
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.optimize import minimize, least_squares
from .forward_kinematics import ForwardKinematics


class InverseKinematics:
    """
    Inverse Kinematics solver using numerical optimization methods
    Supports multiple solving methods: CCD, Jacobian, and numerical optimization
    """

    def __init__(self, forward_kinematics: ForwardKinematics = None):
        """
        Initialize inverse kinematics solver

        Args:
            forward_kinematics: Forward kinematics solver instance
        """
        self.fk = forward_kinematics or ForwardKinematics()
        self.num_joints = self.fk.num_joints

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        method: str = 'optimization',
        max_iterations: int = 1000,
        tolerance: float = 1e-4
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve inverse kinematics

        Args:
            target_position: Desired end-effector position [x, y, z]
            target_orientation: Desired orientation (3x3 rotation matrix), optional
            initial_guess: Initial joint angles, if None uses random
            method: Solving method ('optimization', 'jacobian', 'ccd')
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Tuple of (joint_angles, success, error)
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.num_joints)

        if method == 'optimization':
            return self._solve_optimization(
                target_position, target_orientation, initial_guess, tolerance
            )
        elif method == 'jacobian':
            return self._solve_jacobian(
                target_position, target_orientation, initial_guess,
                max_iterations, tolerance
            )
        elif method == 'ccd':
            return self._solve_ccd(
                target_position, initial_guess, max_iterations, tolerance
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def _solve_optimization(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        initial_guess: np.ndarray,
        tolerance: float
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK using scipy optimization
        """
        def objective(joint_angles):
            pos, rot = self.fk.compute(joint_angles)

            # Position error
            pos_error = np.linalg.norm(pos - target_position)

            # Orientation error (if provided)
            if target_orientation is not None:
                # Frobenius norm of rotation difference
                rot_error = np.linalg.norm(rot - target_orientation, 'fro')
                return pos_error + 0.5 * rot_error

            return pos_error

        # Joint limits (typical for industrial robots)
        bounds = [(-np.pi, np.pi) for _ in range(self.num_joints)]

        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            options={'ftol': tolerance, 'maxiter': 500}
        )

        error = objective(result.x)
        success = error < tolerance * 10

        return result.x, success, error

    def _solve_jacobian(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        initial_guess: np.ndarray,
        max_iterations: int,
        tolerance: float
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK using Jacobian-based iterative method (damped least squares)
        """
        joint_angles = initial_guess.copy()
        lambda_damping = 0.1  # Damping factor

        for iteration in range(max_iterations):
            pos, rot = self.fk.compute(joint_angles)

            # Position error
            pos_error = target_position - pos

            # Check convergence
            error_magnitude = np.linalg.norm(pos_error)
            if error_magnitude < tolerance:
                return joint_angles, True, error_magnitude

            # Compute Jacobian
            J = self.fk.jacobian(joint_angles)

            # Use only position part if no orientation specified
            if target_orientation is None:
                J_pos = J[:3, :]
                error_vector = pos_error
            else:
                # Include orientation error
                rot_error = self._rotation_error(rot, target_orientation)
                error_vector = np.concatenate([pos_error, rot_error])
                J_pos = J

            # Damped least squares
            J_T = J_pos.T
            delta_theta = J_T @ np.linalg.inv(
                J_pos @ J_T + lambda_damping**2 * np.eye(J_pos.shape[0])
            ) @ error_vector

            # Update joint angles
            joint_angles += delta_theta * 0.5  # Step size factor

            # Enforce joint limits
            joint_angles = np.clip(joint_angles, -np.pi, np.pi)

        # Final error check
        pos, _ = self.fk.compute(joint_angles)
        final_error = np.linalg.norm(target_position - pos)

        return joint_angles, final_error < tolerance * 10, final_error

    def _solve_ccd(
        self,
        target_position: np.ndarray,
        initial_guess: np.ndarray,
        max_iterations: int,
        tolerance: float
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK using Cyclic Coordinate Descent (CCD) method
        Fast and simple method, works well for position-only targets
        """
        joint_angles = initial_guess.copy()

        for iteration in range(max_iterations):
            # Check convergence
            pos, _ = self.fk.compute(joint_angles)
            error = np.linalg.norm(target_position - pos)

            if error < tolerance:
                return joint_angles, True, error

            # Iterate through joints in reverse order
            for i in range(self.num_joints - 1, -1, -1):
                # Get current joint position and end-effector position
                joint_positions = self.fk.get_joint_positions(joint_angles)
                joint_pos = joint_positions[i]
                end_effector_pos, _ = self.fk.compute(joint_angles)

                # Vectors from joint to end-effector and target
                to_end = end_effector_pos - joint_pos
                to_target = target_position - joint_pos

                # Skip if vectors are too small
                if np.linalg.norm(to_end) < 1e-6 or np.linalg.norm(to_target) < 1e-6:
                    continue

                # Normalize vectors
                to_end = to_end / np.linalg.norm(to_end)
                to_target = to_target / np.linalg.norm(to_target)

                # Compute angle between vectors
                dot = np.clip(np.dot(to_end, to_target), -1.0, 1.0)
                angle = np.arccos(dot)

                # Determine rotation direction
                cross = np.cross(to_end, to_target)

                # Get axis of rotation for this joint (simplified - assumes Z-axis)
                if np.linalg.norm(cross) > 1e-6:
                    if cross[2] < 0:
                        angle = -angle

                # Update joint angle
                joint_angles[i] += angle * 0.5  # Damping factor

                # Enforce joint limits
                joint_angles[i] = np.clip(joint_angles[i], -np.pi, np.pi)

        # Final error
        pos, _ = self.fk.compute(joint_angles)
        final_error = np.linalg.norm(target_position - pos)

        return joint_angles, final_error < tolerance * 10, final_error

    def _rotation_error(self, R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
        """
        Compute orientation error as axis-angle representation
        """
        R_error = R_target @ R_current.T
        angle = np.arccos((np.trace(R_error) - 1) / 2)

        if np.abs(angle) < 1e-6:
            return np.zeros(3)

        axis = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) / (2 * np.sin(angle))

        return axis * angle

    def solve_multiple_targets(
        self,
        targets: List[np.ndarray],
        method: str = 'optimization',
        **kwargs
    ) -> List[Tuple[np.ndarray, bool, float]]:
        """
        Solve IK for multiple target positions sequentially

        Args:
            targets: List of target positions
            method: Solving method
            **kwargs: Additional arguments for solve()

        Returns:
            List of (joint_angles, success, error) tuples
        """
        results = []
        current_angles = np.zeros(self.num_joints)

        for target in targets:
            joint_angles, success, error = self.solve(
                target,
                initial_guess=current_angles,
                method=method,
                **kwargs
            )
            results.append((joint_angles, success, error))
            current_angles = joint_angles

        return results

    def is_reachable(
        self,
        target_position: np.ndarray,
        tolerance: float = 0.01
    ) -> bool:
        """
        Check if a target position is reachable

        Args:
            target_position: Target position
            tolerance: Acceptable error

        Returns:
            True if reachable
        """
        # Try multiple initial guesses
        initial_guesses = [
            np.zeros(self.num_joints),
            np.random.uniform(-np.pi/2, np.pi/2, self.num_joints),
            np.random.uniform(-np.pi, np.pi, self.num_joints)
        ]

        for guess in initial_guesses:
            _, success, error = self.solve(
                target_position,
                initial_guess=guess,
                method='optimization',
                tolerance=tolerance
            )
            if success:
                return True

        return False
