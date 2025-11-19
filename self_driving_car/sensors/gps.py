"""
GPS sensor for the self-driving car
"""
import numpy as np


class GPS:
    """GPS sensor for position tracking"""

    def __init__(self, noise_std=0.1):
        """
        Initialize GPS sensor

        Args:
            noise_std: Standard deviation of GPS noise in meters
        """
        self.noise_std = noise_std

    def get_position(self, true_position):
        """
        Get GPS position with noise

        Args:
            true_position: True position [x, y, z]

        Returns:
            Noisy GPS position
        """
        noise = np.random.normal(0, self.noise_std, 3)
        gps_position = np.array(true_position) + noise

        return gps_position.tolist()

    def get_accuracy(self):
        """Get GPS accuracy estimate"""
        return self.noise_std
