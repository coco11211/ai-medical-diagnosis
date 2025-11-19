"""
Path Planning Module
Generates smooth trajectories between waypoints using various interpolation methods
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum
from scipy.interpolate import CubicSpline, interp1d
from dataclasses import dataclass


class InterpolationType(Enum):
    """Trajectory interpolation types"""
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    QUINTIC = "quintic"
    TRAPEZOIDAL = "trapezoidal"


@dataclass
class Waypoint:
    """Waypoint definition"""
    position: np.ndarray  # 3D position [x, y, z]
    orientation: Optional[np.ndarray] = None  # 3x3 rotation matrix
    velocity: float = 0.0  # Desired velocity at waypoint
    time: Optional[float] = None  # Time to reach waypoint


class PathPlanner:
    """
    Path planner for generating smooth trajectories
    """

    def __init__(
        self,
        interpolation_type: InterpolationType = InterpolationType.CUBIC_SPLINE,
        max_velocity: float = 1.0,
        max_acceleration: float = 2.0,
        time_step: float = 0.01
    ):
        """
        Initialize path planner

        Args:
            interpolation_type: Type of interpolation to use
            max_velocity: Maximum velocity (m/s)
            max_acceleration: Maximum acceleration (m/s²)
            time_step: Time step for trajectory generation (seconds)
        """
        self.interpolation_type = interpolation_type
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.time_step = time_step

    def plan_path(
        self,
        waypoints: List[Waypoint],
        num_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan a path through waypoints

        Args:
            waypoints: List of waypoints to visit
            num_points: Number of points in trajectory (if None, auto-calculated)

        Returns:
            Tuple of (positions, velocities, times)
            - positions: Array of shape (N, 3) with positions
            - velocities: Array of shape (N, 3) with velocities
            - times: Array of shape (N,) with time stamps
        """
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required")

        # Extract waypoint positions
        waypoint_positions = np.array([wp.position for wp in waypoints])

        # Calculate times if not specified
        times = self._calculate_waypoint_times(waypoints)

        # Determine number of points
        if num_points is None:
            total_time = times[-1]
            num_points = int(total_time / self.time_step) + 1

        # Generate trajectory based on interpolation type
        if self.interpolation_type == InterpolationType.LINEAR:
            positions, velocities, time_stamps = self._linear_interpolation(
                waypoint_positions, times, num_points
            )
        elif self.interpolation_type == InterpolationType.CUBIC_SPLINE:
            positions, velocities, time_stamps = self._cubic_spline_interpolation(
                waypoint_positions, times, num_points
            )
        elif self.interpolation_type == InterpolationType.QUINTIC:
            positions, velocities, time_stamps = self._quintic_interpolation(
                waypoint_positions, times, num_points
            )
        elif self.interpolation_type == InterpolationType.TRAPEZOIDAL:
            positions, velocities, time_stamps = self._trapezoidal_velocity_profile(
                waypoint_positions, times, num_points
            )
        else:
            raise ValueError(f"Unknown interpolation type: {self.interpolation_type}")

        return positions, velocities, time_stamps

    def _calculate_waypoint_times(self, waypoints: List[Waypoint]) -> np.ndarray:
        """
        Calculate time to reach each waypoint based on distance and velocity limits
        """
        times = [0.0]
        current_time = 0.0

        for i in range(1, len(waypoints)):
            if waypoints[i].time is not None:
                # Use specified time
                current_time = waypoints[i].time
            else:
                # Calculate based on distance and max velocity
                distance = np.linalg.norm(
                    waypoints[i].position - waypoints[i-1].position
                )
                time_for_segment = distance / self.max_velocity
                current_time += time_for_segment

            times.append(current_time)

        return np.array(times)

    def _linear_interpolation(
        self,
        waypoint_positions: np.ndarray,
        times: np.ndarray,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linear interpolation between waypoints"""
        time_stamps = np.linspace(times[0], times[-1], num_points)

        # Interpolate each dimension
        positions = np.zeros((num_points, 3))
        for dim in range(3):
            interpolator = interp1d(
                times, waypoint_positions[:, dim],
                kind='linear', fill_value='extrapolate'
            )
            positions[:, dim] = interpolator(time_stamps)

        # Calculate velocities (finite differences)
        velocities = np.zeros((num_points, 3))
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * self.time_step)
        velocities[0] = (positions[1] - positions[0]) / self.time_step
        velocities[-1] = (positions[-1] - positions[-2]) / self.time_step

        return positions, velocities, time_stamps

    def _cubic_spline_interpolation(
        self,
        waypoint_positions: np.ndarray,
        times: np.ndarray,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cubic spline interpolation for smooth trajectories"""
        time_stamps = np.linspace(times[0], times[-1], num_points)

        positions = np.zeros((num_points, 3))
        velocities = np.zeros((num_points, 3))

        # Spline for each dimension
        for dim in range(3):
            cs = CubicSpline(times, waypoint_positions[:, dim], bc_type='natural')
            positions[:, dim] = cs(time_stamps)
            velocities[:, dim] = cs(time_stamps, 1)  # First derivative

        return positions, velocities, time_stamps

    def _quintic_interpolation(
        self,
        waypoint_positions: np.ndarray,
        times: np.ndarray,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quintic (5th order) polynomial interpolation
        Ensures continuity in position, velocity, and acceleration
        """
        time_stamps = np.linspace(times[0], times[-1], num_points)
        positions = np.zeros((num_points, 3))
        velocities = np.zeros((num_points, 3))

        # Process each segment
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            p0, p1 = waypoint_positions[i], waypoint_positions[i + 1]

            # Time indices for this segment
            mask = (time_stamps >= t0) & (time_stamps <= t1)
            segment_times = time_stamps[mask]

            # Normalize time to [0, 1]
            tau = (segment_times - t0) / (t1 - t0)

            # Quintic polynomial coefficients (assuming zero vel/acc at waypoints)
            a0 = p0
            a1 = np.zeros(3)
            a2 = np.zeros(3)
            a3 = 10 * (p1 - p0)
            a4 = -15 * (p1 - p0)
            a5 = 6 * (p1 - p0)

            # Position: p(τ) = a0 + a1*τ + a2*τ² + a3*τ³ + a4*τ⁴ + a5*τ⁵
            tau_matrix = np.array([
                np.ones_like(tau),
                tau,
                tau**2,
                tau**3,
                tau**4,
                tau**5
            ]).T

            coeffs = np.array([a0, a1, a2, a3, a4, a5])
            positions[mask] = tau_matrix @ coeffs

            # Velocity: v(τ) = (a1 + 2*a2*τ + 3*a3*τ² + 4*a4*τ³ + 5*a5*τ⁴) / (t1-t0)
            vel_tau_matrix = np.array([
                np.zeros_like(tau),
                np.ones_like(tau),
                2*tau,
                3*tau**2,
                4*tau**3,
                5*tau**4
            ]).T

            velocities[mask] = (vel_tau_matrix @ coeffs) / (t1 - t0)

        return positions, velocities, time_stamps

    def _trapezoidal_velocity_profile(
        self,
        waypoint_positions: np.ndarray,
        times: np.ndarray,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trajectory with trapezoidal velocity profile
        (acceleration -> constant velocity -> deceleration)
        """
        time_stamps = np.linspace(times[0], times[-1], num_points)
        positions = np.zeros((num_points, 3))
        velocities = np.zeros((num_points, 3))

        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            p0, p1 = waypoint_positions[i], waypoint_positions[i + 1]

            mask = (time_stamps >= t0) & (time_stamps <= t1)
            segment_times = time_stamps[mask]

            if len(segment_times) == 0:
                continue

            # Direction vector
            direction = p1 - p0
            distance = np.linalg.norm(direction)
            direction = direction / (distance + 1e-10)

            # Time for acceleration/deceleration phases
            t_accel = min(self.max_velocity / self.max_acceleration, (t1 - t0) / 2)
            t_decel = t_accel

            # Normalized time
            t_norm = segment_times - t0
            dt = t1 - t0

            for j, t in enumerate(t_norm):
                if t < t_accel:
                    # Acceleration phase
                    v = self.max_acceleration * t
                    s = 0.5 * self.max_acceleration * t**2
                elif t < dt - t_decel:
                    # Constant velocity phase
                    v = self.max_velocity
                    s = (0.5 * self.max_acceleration * t_accel**2 +
                         self.max_velocity * (t - t_accel))
                else:
                    # Deceleration phase
                    t_dec = t - (dt - t_decel)
                    v = self.max_velocity - self.max_acceleration * t_dec
                    s = (0.5 * self.max_acceleration * t_accel**2 +
                         self.max_velocity * (dt - t_accel - t_decel) +
                         self.max_velocity * t_dec -
                         0.5 * self.max_acceleration * t_dec**2)

                # Scale to actual distance
                s_normalized = s / (0.5 * self.max_acceleration * t_accel**2 +
                                   self.max_velocity * (dt - t_accel - t_decel) +
                                   0.5 * self.max_velocity * t_decel)
                s_normalized = np.clip(s_normalized, 0, 1)

                positions[mask][j] = p0 + direction * distance * s_normalized
                velocities[mask][j] = direction * v

        return positions, velocities, time_stamps

    def smooth_path(
        self,
        path: np.ndarray,
        window_size: int = 5,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Smooth a path using moving average filter

        Args:
            path: Array of shape (N, 3) with positions
            window_size: Size of smoothing window
            iterations: Number of smoothing iterations

        Returns:
            Smoothed path
        """
        smoothed = path.copy()

        for _ in range(iterations):
            for i in range(window_size // 2, len(path) - window_size // 2):
                smoothed[i] = np.mean(
                    smoothed[i - window_size//2:i + window_size//2 + 1],
                    axis=0
                )

        return smoothed
