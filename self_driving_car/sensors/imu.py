"""
IMU (Inertial Measurement Unit) sensor for the self-driving car
"""
import numpy as np
import pybullet as p


class IMU:
    """IMU sensor for orientation and acceleration tracking"""

    def __init__(self, accel_noise_std=0.01, gyro_noise_std=0.001):
        """
        Initialize IMU sensor

        Args:
            accel_noise_std: Accelerometer noise standard deviation
            gyro_noise_std: Gyroscope noise standard deviation
        """
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.last_velocity = np.array([0.0, 0.0, 0.0])

    def get_measurements(self, car_id):
        """
        Get IMU measurements

        Args:
            car_id: PyBullet car object ID

        Returns:
            Dictionary with orientation, angular velocity, and acceleration
        """
        # Get position and orientation
        pos, orn = p.getBasePositionAndOrientation(car_id)

        # Get velocities
        linear_vel, angular_vel = p.getBaseVelocity(car_id)

        # Calculate acceleration (simple derivative)
        linear_vel_array = np.array(linear_vel)
        acceleration = (linear_vel_array - self.last_velocity) * 240  # Assuming 240 Hz
        self.last_velocity = linear_vel_array

        # Add noise
        accel_noise = np.random.normal(0, self.accel_noise_std, 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_std, 3)

        noisy_acceleration = acceleration + accel_noise
        noisy_angular_vel = np.array(angular_vel) + gyro_noise

        # Convert quaternion to euler angles
        euler = p.getEulerFromQuaternion(orn)

        return {
            'orientation': euler,  # roll, pitch, yaw
            'quaternion': orn,
            'angular_velocity': noisy_angular_vel.tolist(),
            'linear_acceleration': noisy_acceleration.tolist(),
            'linear_velocity': linear_vel
        }

    def get_heading(self, car_id):
        """
        Get car heading (yaw angle)

        Args:
            car_id: PyBullet car object ID

        Returns:
            Heading angle in radians
        """
        _, orn = p.getBasePositionAndOrientation(car_id)
        euler = p.getEulerFromQuaternion(orn)
        return euler[2]  # yaw
