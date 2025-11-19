"""
Forward Kinematics Module
Computes end-effector position and orientation from joint angles using DH parameters
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class DHParameter:
    """Denavit-Hartenberg parameters for a joint"""
    a: float  # Link length
    alpha: float  # Link twist
    d: float  # Link offset
    theta: float  # Joint angle (variable for revolute joints)


class ForwardKinematics:
    """
    Forward Kinematics solver using Denavit-Hartenberg convention
    """

    def __init__(self, dh_params: List[DHParameter] = None):
        """
        Initialize forward kinematics solver

        Args:
            dh_params: List of DH parameters for each joint
        """
        self.dh_params = dh_params or self._get_default_dh_params()
        self.num_joints = len(self.dh_params)

    def _get_default_dh_params(self) -> List[DHParameter]:
        """
        Get default DH parameters for a 6-DOF robot arm
        (Similar to UR5/industrial robot configuration)
        """
        return [
            DHParameter(a=0, alpha=np.pi/2, d=0.089159, theta=0),
            DHParameter(a=-0.425, alpha=0, d=0, theta=0),
            DHParameter(a=-0.39225, alpha=0, d=0, theta=0),
            DHParameter(a=0, alpha=np.pi/2, d=0.10915, theta=0),
            DHParameter(a=0, alpha=-np.pi/2, d=0.09465, theta=0),
            DHParameter(a=0, alpha=0, d=0.0823, theta=0),
        ]

    def dh_matrix(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        Compute DH transformation matrix

        Args:
            a: Link length
            alpha: Link twist
            d: Link offset
            theta: Joint angle

        Returns:
            4x4 transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def compute(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics

        Args:
            joint_angles: Array of joint angles (radians)

        Returns:
            Tuple of (position, orientation_matrix)
            - position: [x, y, z] end-effector position
            - orientation_matrix: 3x3 rotation matrix
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")

        # Start with identity matrix
        T = np.eye(4)

        # Multiply transformation matrices
        for i, (dh, angle) in enumerate(zip(self.dh_params, joint_angles)):
            # Update theta with joint angle
            theta = dh.theta + angle
            T_i = self.dh_matrix(dh.a, dh.alpha, dh.d, theta)
            T = T @ T_i

        # Extract position and orientation
        position = T[:3, 3]
        orientation = T[:3, :3]

        return position, orientation

    def compute_all_transforms(self, joint_angles: np.ndarray) -> List[np.ndarray]:
        """
        Compute transformation matrices for all joints (for visualization)

        Args:
            joint_angles: Array of joint angles

        Returns:
            List of 4x4 transformation matrices for each joint
        """
        transforms = []
        T = np.eye(4)

        for dh, angle in zip(self.dh_params, joint_angles):
            theta = dh.theta + angle
            T_i = self.dh_matrix(dh.a, dh.alpha, dh.d, theta)
            T = T @ T_i
            transforms.append(T.copy())

        return transforms

    def get_joint_positions(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Get positions of all joints in 3D space

        Args:
            joint_angles: Array of joint angles

        Returns:
            Array of shape (num_joints+1, 3) with positions
        """
        transforms = self.compute_all_transforms(joint_angles)
        positions = [np.array([0, 0, 0])]  # Base position

        for T in transforms:
            positions.append(T[:3, 3])

        return np.array(positions)

    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix (velocity mapping from joint to Cartesian space)

        Args:
            joint_angles: Current joint angles

        Returns:
            6xN Jacobian matrix (3 for linear, 3 for angular velocity)
        """
        epsilon = 1e-6
        J = np.zeros((6, self.num_joints))

        # Current end-effector pose
        pos0, rot0 = self.compute(joint_angles)

        for i in range(self.num_joints):
            # Perturb joint angle
            angles_plus = joint_angles.copy()
            angles_plus[i] += epsilon

            pos_plus, rot_plus = self.compute(angles_plus)

            # Linear velocity (position derivative)
            J[:3, i] = (pos_plus - pos0) / epsilon

            # Angular velocity (orientation derivative)
            # Approximate from rotation matrix change
            rot_diff = rot_plus @ rot0.T
            axis_angle = self._rotation_matrix_to_axis_angle(rot_diff)
            J[3:, i] = axis_angle / epsilon

        return J

    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation"""
        angle = np.arccos((np.trace(R) - 1) / 2)
        if np.abs(angle) < 1e-6:
            return np.zeros(3)

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))

        return axis * angle
