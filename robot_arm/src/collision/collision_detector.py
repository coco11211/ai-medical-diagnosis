"""
Collision Detection Module
Detects and avoids collisions between robot and obstacles
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .obstacle import Obstacle
from ..kinematics.forward_kinematics import ForwardKinematics


class CollisionDetector:
    """
    Collision detection and avoidance for robot arm
    """

    def __init__(
        self,
        forward_kinematics: Optional[ForwardKinematics] = None,
        link_radii: Optional[List[float]] = None,
        safety_margin: float = 0.05
    ):
        """
        Initialize collision detector

        Args:
            forward_kinematics: FK solver for robot
            link_radii: Radius of each robot link (for cylinder approximation)
            safety_margin: Additional safety margin in meters
        """
        self.fk = forward_kinematics or ForwardKinematics()
        self.link_radii = link_radii or [0.05] * self.fk.num_joints  # Default 5cm radius
        self.safety_margin = safety_margin
        self.obstacles: List[Obstacle] = []

    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the environment"""
        self.obstacles.append(obstacle)

    def remove_obstacle(self, index: int):
        """Remove an obstacle by index"""
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)

    def clear_obstacles(self):
        """Clear all obstacles"""
        self.obstacles.clear()

    def check_collision(
        self,
        joint_angles: np.ndarray,
        check_self_collision: bool = False
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if robot configuration collides with obstacles

        Args:
            joint_angles: Joint angles to check
            check_self_collision: Also check for self-collision

        Returns:
            Tuple of (collision_detected, collision_info)
        """
        # Get robot link positions
        joint_positions = self.fk.get_joint_positions(joint_angles)

        collisions = []

        # Check collision for each link
        for i in range(len(joint_positions) - 1):
            p1 = joint_positions[i]
            p2 = joint_positions[i + 1]
            radius = self.link_radii[i] + self.safety_margin

            # Check against all obstacles
            for obs_idx, obstacle in enumerate(self.obstacles):
                # Check link-obstacle collision
                dist = self._line_segment_to_obstacle_distance(p1, p2, obstacle)

                if dist < radius:
                    collisions.append({
                        'link': i,
                        'obstacle': obs_idx,
                        'distance': dist,
                        'penetration': radius - dist
                    })

        # Self-collision check
        if check_self_collision:
            self_collisions = self._check_self_collision(joint_positions)
            collisions.extend(self_collisions)

        return len(collisions) > 0, collisions

    def check_trajectory_collision(
        self,
        joint_trajectory: np.ndarray,
        check_interval: int = 1
    ) -> Tuple[bool, List[int]]:
        """
        Check if a trajectory has collisions

        Args:
            joint_trajectory: Array of shape (N, num_joints)
            check_interval: Check every N-th point (for speed)

        Returns:
            Tuple of (has_collision, collision_indices)
        """
        collision_indices = []

        for i in range(0, len(joint_trajectory), check_interval):
            has_collision, _ = self.check_collision(joint_trajectory[i])
            if has_collision:
                collision_indices.append(i)

        return len(collision_indices) > 0, collision_indices

    def get_minimum_distance(
        self,
        joint_angles: np.ndarray
    ) -> Tuple[float, Optional[Dict]]:
        """
        Get minimum distance to nearest obstacle

        Args:
            joint_angles: Joint configuration

        Returns:
            Tuple of (min_distance, closest_pair_info)
        """
        joint_positions = self.fk.get_joint_positions(joint_angles)

        min_distance = float('inf')
        closest_pair = None

        for i in range(len(joint_positions) - 1):
            p1 = joint_positions[i]
            p2 = joint_positions[i + 1]

            for obs_idx, obstacle in enumerate(self.obstacles):
                dist = self._line_segment_to_obstacle_distance(p1, p2, obstacle)

                if dist < min_distance:
                    min_distance = dist
                    closest_pair = {
                        'link': i,
                        'obstacle': obs_idx,
                        'distance': dist
                    }

        return min_distance, closest_pair

    def _line_segment_to_obstacle_distance(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        obstacle: Obstacle
    ) -> float:
        """
        Calculate minimum distance from line segment to obstacle

        Args:
            p1: Start point of line segment
            p2: End point of line segment
            obstacle: Obstacle

        Returns:
            Minimum distance
        """
        # Sample points along line segment
        num_samples = 10
        t = np.linspace(0, 1, num_samples)
        points = p1[:, np.newaxis] + (p2 - p1)[:, np.newaxis] * t

        # Calculate distance for each sample point
        min_dist = float('inf')
        for i in range(num_samples):
            point = points[:, i]
            dist = obstacle.distance_to_point(point)
            min_dist = min(min_dist, dist)

        return min_dist

    def _check_self_collision(
        self,
        joint_positions: np.ndarray
    ) -> List[Dict]:
        """
        Check for self-collision between robot links

        Args:
            joint_positions: Array of joint positions

        Returns:
            List of self-collision information
        """
        collisions = []

        # Check non-adjacent links
        for i in range(len(joint_positions) - 1):
            for j in range(i + 2, len(joint_positions) - 1):
                # Skip adjacent links
                if abs(i - j) <= 1:
                    continue

                # Calculate distance between link segments
                p1_start = joint_positions[i]
                p1_end = joint_positions[i + 1]
                p2_start = joint_positions[j]
                p2_end = joint_positions[j + 1]

                dist = self._line_segment_distance(
                    p1_start, p1_end, p2_start, p2_end
                )

                min_clearance = self.link_radii[i] + self.link_radii[j] + self.safety_margin

                if dist < min_clearance:
                    collisions.append({
                        'link1': i,
                        'link2': j,
                        'distance': dist,
                        'type': 'self_collision'
                    })

        return collisions

    def _line_segment_distance(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray
    ) -> float:
        """
        Calculate minimum distance between two line segments

        Args:
            p1, p2: Points defining first segment
            p3, p4: Points defining second segment

        Returns:
            Minimum distance
        """
        # Sample-based approach for simplicity
        num_samples = 10
        min_dist = float('inf')

        for t1 in np.linspace(0, 1, num_samples):
            point1 = p1 + t1 * (p2 - p1)

            for t2 in np.linspace(0, 1, num_samples):
                point2 = p3 + t2 * (p4 - p3)

                dist = np.linalg.norm(point1 - point2)
                min_dist = min(min_dist, dist)

        return min_dist

    def generate_collision_free_path(
        self,
        start_angles: np.ndarray,
        goal_angles: np.ndarray,
        max_iterations: int = 1000,
        step_size: float = 0.1
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Generate collision-free path using simple RRT-like approach

        Args:
            start_angles: Starting configuration
            goal_angles: Goal configuration
            max_iterations: Maximum planning iterations
            step_size: Step size in joint space (radians)

        Returns:
            Tuple of (path, success)
        """
        # Check if start and goal are collision-free
        start_collision, _ = self.check_collision(start_angles)
        goal_collision, _ = self.check_collision(goal_angles)

        if start_collision or goal_collision:
            print("Start or goal in collision")
            return None, False

        # Simple straight-line path with collision checking
        # Try direct interpolation first
        num_points = int(np.linalg.norm(goal_angles - start_angles) / step_size) + 2
        direct_path = np.linspace(start_angles, goal_angles, num_points)

        has_collision, _ = self.check_trajectory_collision(direct_path)

        if not has_collision:
            return direct_path, True

        # If direct path has collision, use simple planning
        # This is a simplified version - full RRT/RRT* would be better
        print("Direct path has collision, attempting simple avoidance...")

        # Try adding intermediate waypoints with random perturbations
        for attempt in range(10):
            # Create waypoint with random perturbation
            mid_angles = (start_angles + goal_angles) / 2
            perturbation = np.random.uniform(-np.pi/4, np.pi/4, len(mid_angles))
            waypoint = mid_angles + perturbation

            # Check waypoint collision
            waypoint_collision, _ = self.check_collision(waypoint)
            if waypoint_collision:
                continue

            # Create path through waypoint
            path1 = np.linspace(start_angles, waypoint, num_points // 2)
            path2 = np.linspace(waypoint, goal_angles, num_points // 2)
            combined_path = np.vstack([path1, path2])

            # Check combined path
            has_collision, _ = self.check_trajectory_collision(combined_path)

            if not has_collision:
                return combined_path, True

        print("Failed to find collision-free path")
        return None, False

    def compute_repulsive_force(
        self,
        joint_angles: np.ndarray,
        influence_distance: float = 0.2
    ) -> np.ndarray:
        """
        Compute repulsive force from obstacles (for potential field planning)

        Args:
            joint_angles: Current joint configuration
            influence_distance: Distance at which obstacles exert force

        Returns:
            Repulsive force vector in joint space
        """
        joint_positions = self.fk.get_joint_positions(joint_angles)
        repulsive_force = np.zeros(len(joint_angles))

        for i in range(len(joint_positions) - 1):
            p1 = joint_positions[i]
            p2 = joint_positions[i + 1]
            link_center = (p1 + p2) / 2

            for obstacle in self.obstacles:
                dist = obstacle.distance_to_point(link_center)

                if 0 < dist < influence_distance:
                    # Repulsive force magnitude (inversely proportional to distance)
                    force_magnitude = (1.0 / dist - 1.0 / influence_distance)

                    # Direction away from obstacle
                    direction = (link_center - obstacle.position)
                    direction = direction / (np.linalg.norm(direction) + 1e-10)

                    # Apply force to joint (simplified - should use Jacobian)
                    repulsive_force[i] += force_magnitude * 0.1

        return repulsive_force

    def visualize_obstacles(self) -> List[Dict]:
        """
        Get obstacle information for visualization

        Returns:
            List of obstacle visualization data
        """
        vis_data = []

        for i, obstacle in enumerate(self.obstacles):
            data = {
                'id': i,
                'type': obstacle.obstacle_type.value,
                'position': obstacle.position.tolist(),
                'dimensions': obstacle.dimensions,
                'orientation': obstacle.orientation.tolist()
            }
            vis_data.append(data)

        return vis_data
