"""
Car model with integrated sensors
"""
import pybullet as p
import numpy as np
from .camera import Camera
from .lidar import LIDAR
from .gps import GPS
from .imu import IMU


class SelfDrivingCar:
    """Self-driving car with multiple sensors"""

    def __init__(self, start_position=[0, 0, 0.5], start_orientation=[0, 0, 0, 1]):
        """
        Initialize the self-driving car

        Args:
            start_position: Initial position [x, y, z]
            start_orientation: Initial orientation quaternion
        """
        self.start_position = start_position
        self.start_orientation = start_orientation
        self.car_id = None

        # Initialize sensors
        self.camera = Camera(width=640, height=480)
        self.lidar = LIDAR(num_rays=360, max_range=50)
        self.gps = GPS(noise_std=0.1)
        self.imu = IMU()

        # Control parameters
        self.max_force = 20
        self.steering_angle = 0
        self.target_velocity = 0

    def create(self):
        """Create the car in PyBullet"""
        # Car dimensions
        car_length = 2.0
        car_width = 1.0
        car_height = 0.5

        # Create car body
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[car_length/2, car_width/2, car_height/2]
        )

        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[car_length/2, car_width/2, car_height/2],
            rgbaColor=[0, 0, 1, 1]  # Blue car
        )

        self.car_id = p.createMultiBody(
            baseMass=1500,  # 1500 kg
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.start_position,
            baseOrientation=self.start_orientation
        )

        # Add wheels (simplified - visual only)
        self._add_wheels(car_length, car_width, car_height)

        return self.car_id

    def _add_wheels(self, car_length, car_width, car_height):
        """Add visual wheels to the car"""
        wheel_radius = 0.3
        wheel_width = 0.2

        wheel_positions = [
            [car_length/3, car_width/2 + wheel_width/2, -car_height/2],
            [car_length/3, -car_width/2 - wheel_width/2, -car_height/2],
            [-car_length/3, car_width/2 + wheel_width/2, -car_height/2],
            [-car_length/3, -car_width/2 - wheel_width/2, -car_height/2],
        ]

        for pos in wheel_positions:
            wheel_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=wheel_radius,
                length=wheel_width,
                rgbaColor=[0.1, 0.1, 0.1, 1]
            )

            wheel_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=wheel_visual,
                basePosition=[
                    self.start_position[0] + pos[0],
                    self.start_position[1] + pos[1],
                    self.start_position[2] + pos[2]
                ]
            )

    def get_position(self):
        """Get car position"""
        if self.car_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.car_id)
            return pos
        return None

    def get_orientation(self):
        """Get car orientation"""
        if self.car_id is not None:
            _, orn = p.getBasePositionAndOrientation(self.car_id)
            return orn
        return None

    def get_velocity(self):
        """Get car velocity"""
        if self.car_id is not None:
            vel, _ = p.getBaseVelocity(self.car_id)
            return vel
        return None

    def apply_control(self, throttle, steering):
        """
        Apply control commands to the car

        Args:
            throttle: Throttle value [-1, 1] (negative for brake/reverse)
            steering: Steering angle in radians
        """
        if self.car_id is None:
            return

        # Get current orientation
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        # Calculate force direction
        force_magnitude = throttle * self.max_force
        force_x = force_magnitude * np.cos(yaw + steering)
        force_y = force_magnitude * np.sin(yaw + steering)

        # Apply force
        p.applyExternalForce(
            self.car_id,
            -1,
            [force_x, force_y, 0],
            pos,
            p.WORLD_FRAME
        )

        # Apply steering (torque around z-axis)
        steering_torque = steering * 2.0
        p.applyExternalTorque(
            self.car_id,
            -1,
            [0, 0, steering_torque],
            p.WORLD_FRAME
        )

    def get_camera_image(self):
        """Get image from car camera"""
        pos = self.get_position()
        orn = self.get_orientation()
        if pos and orn:
            return self.camera.capture(pos, orn)
        return None, None, None

    def get_lidar_scan(self):
        """Get LIDAR scan data"""
        pos = self.get_position()
        orn = self.get_orientation()
        if pos and orn:
            return self.lidar.scan(pos, orn)
        return []

    def get_gps_reading(self):
        """Get GPS position reading"""
        pos = self.get_position()
        if pos:
            return self.gps.get_position(pos)
        return None

    def get_imu_data(self):
        """Get IMU measurements"""
        if self.car_id is not None:
            return self.imu.get_measurements(self.car_id)
        return None

    def get_all_sensor_data(self):
        """Get data from all sensors"""
        return {
            'camera': self.get_camera_image(),
            'lidar': self.get_lidar_scan(),
            'gps': self.get_gps_reading(),
            'imu': self.get_imu_data(),
            'position': self.get_position(),
            'orientation': self.get_orientation(),
            'velocity': self.get_velocity()
        }

    def reset(self):
        """Reset car to initial position"""
        if self.car_id is not None:
            p.resetBasePositionAndOrientation(
                self.car_id,
                self.start_position,
                self.start_orientation
            )
            p.resetBaseVelocity(self.car_id, [0, 0, 0], [0, 0, 0])
