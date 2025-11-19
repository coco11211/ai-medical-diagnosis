"""
Obstacle avoidance system using potential field method
"""
import numpy as np
import math


class ObstacleAvoidance:
    """Obstacle avoidance using artificial potential fields"""

    def __init__(self, safe_distance=3.0, influence_distance=10.0):
        """
        Initialize obstacle avoidance system

        Args:
            safe_distance: Minimum safe distance from obstacles
            influence_distance: Distance at which obstacles start influencing behavior
        """
        self.safe_distance = safe_distance
        self.influence_distance = influence_distance

    def calculate_repulsive_force(self, car_position, obstacles):
        """
        Calculate repulsive force from obstacles

        Args:
            car_position: Current car position [x, y, z]
            obstacles: List of obstacle data from LIDAR

        Returns:
            Repulsive force vector [fx, fy]
        """
        if not obstacles:
            return np.array([0.0, 0.0])

        repulsive_force = np.array([0.0, 0.0])
        car_pos_2d = np.array([car_position[0], car_position[1]])

        for obstacle in obstacles:
            # Get obstacle position
            if 'hit_position' in obstacle:
                obs_pos = obstacle['hit_position']
                obs_pos_2d = np.array([obs_pos[0], obs_pos[1]])
            else:
                continue

            # Calculate distance
            distance_vector = car_pos_2d - obs_pos_2d
            distance = np.linalg.norm(distance_vector)

            if distance < 0.1:  # Avoid division by zero
                distance = 0.1

            # Calculate repulsive force
            if distance < self.influence_distance:
                # Force magnitude increases as distance decreases
                if distance < self.safe_distance:
                    magnitude = 10.0 / (distance ** 2)  # Strong repulsion
                else:
                    magnitude = 1.0 / (distance ** 2)  # Moderate repulsion

                # Normalize direction and apply magnitude
                direction = distance_vector / distance
                force = magnitude * direction
                repulsive_force += force

        return repulsive_force

    def calculate_avoidance_steering(self, car_position, car_heading, lidar_data):
        """
        Calculate steering angle for obstacle avoidance

        Args:
            car_position: Current car position [x, y, z]
            car_heading: Current heading angle in radians
            lidar_data: LIDAR scan data

        Returns:
            Steering angle in radians
        """
        # Get nearby obstacles
        nearby_obstacles = [
            obs for obs in lidar_data
            if obs['object_id'] >= 0 and obs['distance'] < self.influence_distance
        ]

        if not nearby_obstacles:
            return 0.0

        # Calculate repulsive force
        repulsive_force = self.calculate_repulsive_force(car_position, nearby_obstacles)

        # Convert force to steering angle
        if np.linalg.norm(repulsive_force) < 0.01:
            return 0.0

        # Calculate desired heading
        desired_angle = math.atan2(repulsive_force[1], repulsive_force[0])

        # Calculate steering angle (difference from current heading)
        steering_angle = self._normalize_angle(desired_angle - car_heading)

        # Limit steering angle
        max_steering = 0.5
        steering_angle = np.clip(steering_angle, -max_steering, max_steering)

        return steering_angle

    def check_collision_risk(self, lidar_data, forward_distance=5.0):
        """
        Check if there's a collision risk ahead

        Args:
            lidar_data: LIDAR scan data
            forward_distance: Distance to check ahead

        Returns:
            Boolean indicating collision risk and closest obstacle distance
        """
        if not lidar_data:
            return False, float('inf')

        # Check obstacles in front sector (±30 degrees)
        front_sector = []
        for obs in lidar_data:
            angle = obs['angle']
            # Normalize angle to [-pi, pi]
            angle = self._normalize_angle(angle)

            # Check if in front sector
            if abs(angle) < np.pi / 6:  # ±30 degrees
                if obs['object_id'] >= 0:
                    front_sector.append(obs)

        if not front_sector:
            return False, float('inf')

        # Find closest obstacle in front
        min_distance = min(obs['distance'] for obs in front_sector)

        # Check if too close
        collision_risk = min_distance < forward_distance

        return collision_risk, min_distance

    def calculate_safe_speed(self, lidar_data, current_speed, max_speed=10.0):
        """
        Calculate safe speed based on obstacle proximity

        Args:
            lidar_data: LIDAR scan data
            current_speed: Current speed
            max_speed: Maximum allowed speed

        Returns:
            Safe target speed
        """
        collision_risk, min_distance = self.check_collision_risk(lidar_data)

        if not collision_risk:
            return max_speed

        # Reduce speed based on distance
        if min_distance < self.safe_distance:
            # Emergency braking
            target_speed = 0.0
        elif min_distance < self.influence_distance:
            # Gradual slowdown
            speed_ratio = (min_distance - self.safe_distance) / (self.influence_distance - self.safe_distance)
            target_speed = max_speed * speed_ratio
        else:
            target_speed = max_speed

        return target_speed

    def get_obstacle_map(self, lidar_data, resolution=0.5, range_m=20):
        """
        Create a 2D occupancy grid from LIDAR data

        Args:
            lidar_data: LIDAR scan data
            resolution: Grid resolution in meters
            range_m: Map range in meters

        Returns:
            2D numpy array representing occupancy grid
        """
        grid_size = int(range_m * 2 / resolution)
        grid = np.zeros((grid_size, grid_size))

        center = grid_size // 2

        for obs in lidar_data:
            if obs['object_id'] >= 0:
                # Convert hit position to grid coordinates
                distance = obs['distance']
                angle = obs['angle']

                x = distance * np.cos(angle)
                y = distance * np.sin(angle)

                grid_x = int(center + x / resolution)
                grid_y = int(center + y / resolution)

                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    grid[grid_y, grid_x] = 1

        return grid

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
