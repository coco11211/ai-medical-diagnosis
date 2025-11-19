"""
Obstacle Representation Module
Defines different types of obstacles for collision checking
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class ObstacleType(Enum):
    """Types of obstacles"""
    SPHERE = "sphere"
    BOX = "box"
    CYLINDER = "cylinder"
    MESH = "mesh"


@dataclass
class Obstacle:
    """
    Obstacle representation
    """
    obstacle_type: ObstacleType
    position: np.ndarray  # [x, y, z] center position
    dimensions: Tuple[float, ...]  # Size parameters (depends on type)
    orientation: Optional[np.ndarray] = None  # 3x3 rotation matrix for boxes

    def __post_init__(self):
        if self.orientation is None:
            self.orientation = np.eye(3)

    @classmethod
    def create_sphere(cls, center: np.ndarray, radius: float):
        """Create a spherical obstacle"""
        return cls(
            obstacle_type=ObstacleType.SPHERE,
            position=center,
            dimensions=(radius,)
        )

    @classmethod
    def create_box(
        cls,
        center: np.ndarray,
        size: Tuple[float, float, float],
        orientation: Optional[np.ndarray] = None
    ):
        """Create a box obstacle"""
        return cls(
            obstacle_type=ObstacleType.BOX,
            position=center,
            dimensions=size,
            orientation=orientation or np.eye(3)
        )

    @classmethod
    def create_cylinder(
        cls,
        center: np.ndarray,
        radius: float,
        height: float,
        orientation: Optional[np.ndarray] = None
    ):
        """Create a cylindrical obstacle"""
        return cls(
            obstacle_type=ObstacleType.CYLINDER,
            position=center,
            dimensions=(radius, height),
            orientation=orientation or np.eye(3)
        )

    def distance_to_point(self, point: np.ndarray) -> float:
        """
        Calculate distance from obstacle surface to a point

        Args:
            point: 3D point

        Returns:
            Distance (negative if inside obstacle)
        """
        if self.obstacle_type == ObstacleType.SPHERE:
            radius = self.dimensions[0]
            dist_to_center = np.linalg.norm(point - self.position)
            return dist_to_center - radius

        elif self.obstacle_type == ObstacleType.BOX:
            # Transform point to box local coordinates
            local_point = self.orientation.T @ (point - self.position)

            # Half dimensions
            half_size = np.array(self.dimensions) / 2

            # Distance to box surface
            q = np.abs(local_point) - half_size
            outside_dist = np.linalg.norm(np.maximum(q, 0))
            inside_dist = np.min(np.maximum(q, 0))

            return outside_dist + inside_dist

        elif self.obstacle_type == ObstacleType.CYLINDER:
            radius, height = self.dimensions

            # Transform to local coordinates
            local_point = self.orientation.T @ (point - self.position)

            # Distance in XY plane
            xy_dist = np.linalg.norm(local_point[:2]) - radius

            # Distance along Z axis
            z_dist = abs(local_point[2]) - height / 2

            # Combined distance
            if xy_dist < 0 and z_dist < 0:
                return max(xy_dist, z_dist)  # Inside
            elif xy_dist > 0 and z_dist > 0:
                return np.sqrt(xy_dist**2 + z_dist**2)  # Outside corner
            else:
                return max(xy_dist, z_dist)  # Outside edge

        else:
            raise NotImplementedError(f"Distance calculation not implemented for {self.obstacle_type}")

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside obstacle"""
        return self.distance_to_point(point) < 0

    def get_bounding_sphere(self) -> Tuple[np.ndarray, float]:
        """
        Get bounding sphere for quick collision checks

        Returns:
            Tuple of (center, radius)
        """
        if self.obstacle_type == ObstacleType.SPHERE:
            return self.position, self.dimensions[0]

        elif self.obstacle_type == ObstacleType.BOX:
            # Diagonal of box
            size = np.array(self.dimensions)
            radius = np.linalg.norm(size) / 2
            return self.position, radius

        elif self.obstacle_type == ObstacleType.CYLINDER:
            radius, height = self.dimensions
            bounding_radius = np.sqrt(radius**2 + (height/2)**2)
            return self.position, bounding_radius

        else:
            return self.position, 1.0
