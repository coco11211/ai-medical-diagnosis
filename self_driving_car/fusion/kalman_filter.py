"""
Sensor fusion using Kalman Filter
"""
import numpy as np


class KalmanFilter:
    """Extended Kalman Filter for sensor fusion"""

    def __init__(self, dt=1/240.0):
        """
        Initialize Kalman Filter

        Args:
            dt: Time step in seconds
        """
        self.dt = dt

        # State vector: [x, y, vx, vy, heading, angular_velocity]
        self.state = np.zeros(6)

        # State covariance matrix
        self.P = np.eye(6) * 1.0

        # Process noise covariance
        self.Q = np.eye(6) * 0.1

        # Measurement noise covariance
        self.R_gps = np.eye(2) * 0.5  # GPS position noise
        self.R_imu = np.eye(2) * 0.01  # IMU orientation/velocity noise

        # State transition matrix
        self.F = np.eye(6)
        self.F[0, 2] = dt  # x += vx * dt
        self.F[1, 3] = dt  # y += vy * dt
        self.F[4, 5] = dt  # heading += angular_velocity * dt

    def predict(self):
        """
        Prediction step of Kalman filter

        Returns:
            Predicted state
        """
        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state.copy()

    def update_gps(self, gps_position):
        """
        Update with GPS measurement

        Args:
            gps_position: GPS position [x, y, z]
        """
        # Measurement matrix for GPS (measures x, y)
        H = np.zeros((2, 6))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y

        # Measurement
        z = np.array([gps_position[0], gps_position[1]])

        # Innovation
        y = z - H @ self.state

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gps

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_imu(self, imu_data):
        """
        Update with IMU measurement

        Args:
            imu_data: IMU data dictionary with orientation and velocities
        """
        if imu_data is None:
            return

        # Measurement matrix for IMU (measures heading and angular velocity)
        H = np.zeros((2, 6))
        H[0, 4] = 1  # heading
        H[1, 5] = 1  # angular velocity

        # Measurement
        heading = imu_data['orientation'][2]  # yaw
        angular_vel = imu_data['angular_velocity'][2]  # z-axis rotation

        z = np.array([heading, angular_vel])

        # Innovation
        y = z - H @ self.state

        # Normalize heading error to [-pi, pi]
        y[0] = self._normalize_angle(y[0])

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_imu

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_velocity(self, velocity):
        """
        Update with velocity measurement

        Args:
            velocity: Velocity vector [vx, vy, vz]
        """
        # Measurement matrix for velocity
        H = np.zeros((2, 6))
        H[0, 2] = 1  # vx
        H[1, 3] = 1  # vy

        # Measurement
        z = np.array([velocity[0], velocity[1]])

        # Innovation
        y = z - H @ self.state

        # Innovation covariance
        R = np.eye(2) * 0.1
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_state(self):
        """
        Get current state estimate

        Returns:
            Dictionary with state information
        """
        return {
            'position': [self.state[0], self.state[1], 0],
            'velocity': [self.state[2], self.state[3], 0],
            'heading': self.state[4],
            'angular_velocity': self.state[5],
            'covariance': self.P.copy()
        }

    def get_position_uncertainty(self):
        """Get position uncertainty (standard deviation)"""
        return np.sqrt(self.P[0, 0] + self.P[1, 1])

    def reset(self, initial_state=None):
        """
        Reset filter

        Args:
            initial_state: Initial state vector [x, y, vx, vy, heading, angular_vel]
        """
        if initial_state is not None:
            self.state = np.array(initial_state)
        else:
            self.state = np.zeros(6)

        self.P = np.eye(6) * 1.0

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class SensorFusion:
    """Sensor fusion system combining multiple sensors"""

    def __init__(self):
        """Initialize sensor fusion system"""
        self.kalman_filter = KalmanFilter()
        self.last_update_time = 0

    def update(self, sensor_data):
        """
        Update sensor fusion with new data

        Args:
            sensor_data: Dictionary with sensor measurements

        Returns:
            Fused state estimate
        """
        # Prediction step
        self.kalman_filter.predict()

        # Update with GPS if available
        if sensor_data.get('gps') is not None:
            self.kalman_filter.update_gps(sensor_data['gps'])

        # Update with IMU if available
        if sensor_data.get('imu') is not None:
            self.kalman_filter.update_imu(sensor_data['imu'])

        # Update with velocity if available
        if sensor_data.get('velocity') is not None:
            self.kalman_filter.update_velocity(sensor_data['velocity'])

        return self.kalman_filter.get_state()

    def get_fused_position(self):
        """Get fused position estimate"""
        state = self.kalman_filter.get_state()
        return state['position']

    def get_fused_velocity(self):
        """Get fused velocity estimate"""
        state = self.kalman_filter.get_state()
        return state['velocity']

    def get_fused_heading(self):
        """Get fused heading estimate"""
        state = self.kalman_filter.get_state()
        return state['heading']

    def get_uncertainty(self):
        """Get position uncertainty"""
        return self.kalman_filter.get_position_uncertainty()

    def reset(self):
        """Reset sensor fusion"""
        self.kalman_filter.reset()
