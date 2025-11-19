"""
Particle Filter for robot localization
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class ParticleFilter:
    """
    Particle Filter for Monte Carlo Localization (MCL).
    Used to estimate robot pose given sensor measurements and map.
    """

    def __init__(self,
                 num_particles: int = 1000,
                 grid_size: Tuple[int, int] = (500, 500),
                 initial_pose: Tuple[float, float, float] = None):
        """
        Initialize particle filter.

        Args:
            num_particles: Number of particles
            grid_size: Size of the grid
            initial_pose: Initial pose (x, y, theta), if None distributes randomly
        """
        self.num_particles = num_particles
        self.grid_size = grid_size

        # Initialize particles
        if initial_pose is not None:
            # Initialize around known pose with some noise
            self.particles = np.zeros((num_particles, 3))
            self.particles[:, 0] = initial_pose[0] + np.random.randn(num_particles) * 2
            self.particles[:, 1] = initial_pose[1] + np.random.randn(num_particles) * 2
            self.particles[:, 2] = initial_pose[2] + np.random.randn(num_particles) * 0.1
        else:
            # Random initialization
            self.particles = np.zeros((num_particles, 3))
            self.particles[:, 0] = np.random.uniform(0, grid_size[0], num_particles)
            self.particles[:, 1] = np.random.uniform(0, grid_size[1], num_particles)
            self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)

        # Particle weights (all equal initially)
        self.weights = np.ones(num_particles) / num_particles

        logger.info(f"Particle filter initialized with {num_particles} particles")

    def predict(self, dx: float, dy: float, dtheta: float,
                motion_noise_std: Tuple[float, float, float] = (0.1, 0.1, 0.05)):
        """
        Prediction step: Move particles based on odometry with noise.

        Args:
            dx: Change in x (grid units)
            dy: Change in y (grid units)
            dtheta: Change in orientation (radians)
            motion_noise_std: Standard deviation of motion noise (x, y, theta)
        """
        for i in range(self.num_particles):
            theta = self.particles[i, 2]

            # Add noise to motion
            noisy_dx = dx + np.random.randn() * motion_noise_std[0]
            noisy_dy = dy + np.random.randn() * motion_noise_std[1]
            noisy_dtheta = dtheta + np.random.randn() * motion_noise_std[2]

            # Update particle pose
            self.particles[i, 0] += noisy_dx * np.cos(theta) - noisy_dy * np.sin(theta)
            self.particles[i, 1] += noisy_dx * np.sin(theta) + noisy_dy * np.cos(theta)
            self.particles[i, 2] += noisy_dtheta

            # Normalize angle
            self.particles[i, 2] = np.arctan2(np.sin(self.particles[i, 2]),
                                              np.cos(self.particles[i, 2]))

            # Keep particles within bounds
            self.particles[i, 0] = np.clip(self.particles[i, 0], 0, self.grid_size[0] - 1)
            self.particles[i, 1] = np.clip(self.particles[i, 1], 0, self.grid_size[1] - 1)

    def update(self, sensor_ranges: List[float], sensor_angles: List[float],
               occupancy_grid: np.ndarray, max_range: float = 5.0):
        """
        Update step: Compute particle weights based on sensor measurements.

        Args:
            sensor_ranges: Measured ranges from sensors
            sensor_angles: Angles of sensors relative to robot
            occupancy_grid: Current occupancy grid
            max_range: Maximum sensor range
        """
        for i in range(self.num_particles):
            # Compute likelihood of this particle given sensor measurements
            weight = self._compute_likelihood(
                self.particles[i],
                sensor_ranges,
                sensor_angles,
                occupancy_grid,
                max_range
            )
            self.weights[i] *= weight

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # All weights zero, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles

    def _compute_likelihood(self, particle: np.ndarray,
                          sensor_ranges: List[float],
                          sensor_angles: List[float],
                          occupancy_grid: np.ndarray,
                          max_range: float) -> float:
        """
        Compute likelihood of particle given sensor measurements.
        """
        x, y, theta = particle
        likelihood = 1.0
        sigma = 0.5  # Measurement noise standard deviation

        for measured_range, angle in zip(sensor_ranges, sensor_angles):
            if measured_range <= 0 or measured_range > max_range:
                continue

            # Expected range based on map
            expected_range = self._ray_cast(x, y, theta + angle, occupancy_grid, max_range)

            # Gaussian likelihood
            diff = measured_range - expected_range
            likelihood *= np.exp(-0.5 * (diff / sigma) ** 2)

        return likelihood

    def _ray_cast(self, x: float, y: float, theta: float,
                  occupancy_grid: np.ndarray, max_range: float) -> float:
        """
        Cast a ray from (x, y) at angle theta to find obstacle.
        Returns distance to obstacle or max_range if no obstacle found.
        """
        step = 0.5  # Step size for ray casting
        distance = 0.0

        while distance < max_range:
            # Current position along ray
            rx = int(x + distance * np.cos(theta))
            ry = int(y + distance * np.sin(theta))

            # Check bounds
            if rx < 0 or rx >= occupancy_grid.shape[0] or \
               ry < 0 or ry >= occupancy_grid.shape[1]:
                return max_range

            # Check if obstacle
            if occupancy_grid[rx, ry] >= 100:
                return distance

            distance += step

        return max_range

    def resample(self):
        """
        Resample particles based on weights (low variance resampling).
        """
        # Low variance resampling
        new_particles = np.zeros_like(self.particles)
        r = np.random.uniform(0, 1.0 / self.num_particles)
        c = self.weights[0]
        i = 0

        for m in range(self.num_particles):
            u = r + m / self.num_particles
            while u > c:
                i += 1
                if i >= self.num_particles:
                    i = self.num_particles - 1
                    break
                c += self.weights[i]
            new_particles[m] = self.particles[i]

        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimated_pose(self) -> Tuple[float, float, float]:
        """
        Get estimated pose as weighted average of particles.
        """
        x = np.sum(self.particles[:, 0] * self.weights)
        y = np.sum(self.particles[:, 1] * self.weights)

        # For angle, use circular mean
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        theta = np.arctan2(sin_sum, cos_sum)

        return x, y, theta

    def get_particles(self) -> np.ndarray:
        """Get all particles."""
        return self.particles.copy()

    def get_weights(self) -> np.ndarray:
        """Get particle weights."""
        return self.weights.copy()

    def get_effective_particles(self) -> int:
        """Get effective number of particles (measure of filter health)."""
        return int(1.0 / np.sum(self.weights ** 2))
