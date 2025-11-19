"""
LIDAR sensor for the self-driving car
"""
import pybullet as p
import numpy as np


class LIDAR:
    """LIDAR sensor for distance measurements"""

    def __init__(self, num_rays=360, max_range=50, height_offset=0.5):
        """
        Initialize LIDAR sensor

        Args:
            num_rays: Number of laser rays
            max_range: Maximum detection range in meters
            height_offset: Height above car base
        """
        self.num_rays = num_rays
        self.max_range = max_range
        self.height_offset = height_offset

    def scan(self, car_position, car_orientation):
        """
        Perform LIDAR scan

        Args:
            car_position: Car position [x, y, z]
            car_orientation: Car orientation quaternion

        Returns:
            List of ray hit distances and points
        """
        # Convert quaternion to euler
        euler = p.getEulerFromQuaternion(car_orientation)
        yaw = euler[2]

        # LIDAR position
        lidar_pos = [
            car_position[0],
            car_position[1],
            car_position[2] + self.height_offset
        ]

        # Perform raycasts in a circle
        ray_from = []
        ray_to = []
        angles = np.linspace(0, 2 * np.pi, self.num_rays)

        for angle in angles:
            # Global angle
            global_angle = angle + yaw

            # Ray direction
            ray_end = [
                lidar_pos[0] + self.max_range * np.cos(global_angle),
                lidar_pos[1] + self.max_range * np.sin(global_angle),
                lidar_pos[2]
            ]

            ray_from.append(lidar_pos)
            ray_to.append(ray_end)

        # Perform batch raycast
        results = p.rayTestBatch(ray_from, ray_to)

        # Process results
        scan_data = []
        for i, result in enumerate(results):
            hit_object_id = result[0]
            hit_fraction = result[2]
            hit_position = result[3]

            distance = self.max_range * hit_fraction

            scan_data.append({
                'angle': angles[i],
                'distance': distance,
                'hit_position': hit_position,
                'object_id': hit_object_id
            })

        return scan_data

    def get_point_cloud(self, scan_data):
        """
        Convert scan data to point cloud

        Args:
            scan_data: LIDAR scan data

        Returns:
            Point cloud as numpy array
        """
        points = []
        for data in scan_data:
            if data['object_id'] >= 0:  # Valid hit
                points.append(data['hit_position'])

        return np.array(points)

    def get_obstacles_in_range(self, scan_data, threshold_distance=10):
        """
        Get obstacles within threshold distance

        Args:
            scan_data: LIDAR scan data
            threshold_distance: Distance threshold

        Returns:
            List of nearby obstacles
        """
        obstacles = []
        for data in scan_data:
            if data['object_id'] >= 0 and data['distance'] < threshold_distance:
                obstacles.append(data)

        return obstacles
