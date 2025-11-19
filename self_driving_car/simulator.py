"""
Main Self-Driving Car Simulator
"""
import pybullet as p
import time
import numpy as np
import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_driving_car.environment.world import SimulationWorld
from self_driving_car.sensors.car import SelfDrivingCar
from self_driving_car.perception.lane_detection import LaneDetector
from self_driving_car.control.obstacle_avoidance import ObstacleAvoidance
from self_driving_car.planning.path_planner import AStarPlanner, PotentialFieldPlanner, PathFollower
from self_driving_car.fusion.kalman_filter import SensorFusion


class SelfDrivingCarSimulator:
    """Main simulator class"""

    def __init__(self, gui=True):
        """
        Initialize simulator

        Args:
            gui: Enable GUI mode
        """
        self.gui = gui
        self.world = SimulationWorld(gui=gui)
        self.car = SelfDrivingCar(start_position=[-50, 0, 0.5])

        # Initialize components
        self.lane_detector = LaneDetector()
        self.obstacle_avoidance = ObstacleAvoidance(safe_distance=3.0, influence_distance=10.0)
        self.path_planner = AStarPlanner(grid_resolution=0.5)
        self.potential_field = PotentialFieldPlanner()
        self.path_follower = PathFollower(lookahead_distance=3.0)
        self.sensor_fusion = SensorFusion()

        # Simulation parameters
        self.running = False
        self.autonomous_mode = True
        self.target_speed = 5.0
        self.current_speed = 0.0

        # Path
        self.planned_path = None
        self.goal_position = [50, 0]

        # Visualization
        self.show_camera = True
        self.show_lidar = False

    def setup(self):
        """Setup the simulation"""
        print("Setting up simulation...")

        # Setup world
        self.world.setup()

        # Create car
        self.car.create()

        # Add some dynamic obstacles
        self.world.add_dynamic_obstacle([30, 0, 0.5], velocity=[0, 1, 0])

        print("Simulation setup complete!")
        print("\nControls:")
        print("  Space: Toggle autonomous mode")
        print("  Arrow keys: Manual control (when autonomous mode is off)")
        print("  C: Toggle camera view")
        print("  L: Toggle LIDAR visualization")
        print("  R: Reset simulation")
        print("  Q/ESC: Quit")

        self.running = True

    def run(self):
        """Run the simulation"""
        if not self.running:
            self.setup()

        time_step = 1.0 / 240.0
        p.setTimeStep(time_step)

        frame_count = 0

        try:
            while self.running:
                start_time = time.time()

                # Get sensor data
                sensor_data = self.car.get_all_sensor_data()

                # Sensor fusion
                fused_state = self.sensor_fusion.update(sensor_data)

                # Get camera image and detect lanes
                camera_rgb, camera_depth, camera_seg = self.car.get_camera_image()
                lane_data = None
                if camera_rgb is not None:
                    lane_data = self.lane_detector.detect_lanes(camera_rgb)

                # Get LIDAR data
                lidar_data = self.car.get_lidar_scan()

                # Control logic
                if self.autonomous_mode:
                    throttle, steering = self._autonomous_control(
                        sensor_data,
                        fused_state,
                        lane_data,
                        lidar_data
                    )
                else:
                    throttle, steering = self._manual_control()

                # Apply control
                self.car.apply_control(throttle, steering)

                # Step simulation
                p.stepSimulation()

                # Visualization
                if frame_count % 10 == 0:  # Update display every 10 frames
                    self._update_visualization(camera_rgb, lane_data, lidar_data, fused_state)

                # Handle keyboard input
                self._handle_keyboard()

                # Sleep to maintain real-time
                elapsed = time.time() - start_time
                if elapsed < time_step:
                    time.sleep(time_step - elapsed)

                frame_count += 1

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")

        finally:
            self.cleanup()

    def _autonomous_control(self, sensor_data, fused_state, lane_data, lidar_data):
        """
        Autonomous control logic

        Args:
            sensor_data: Raw sensor data
            fused_state: Fused state from Kalman filter
            lane_data: Lane detection data
            lidar_data: LIDAR scan data

        Returns:
            throttle and steering commands
        """
        # Get current position and heading
        current_pos = fused_state['position']
        current_heading = fused_state['heading']

        # Initialize steering and throttle
        steering = 0.0
        throttle = 0.5

        # 1. Lane following
        lane_steering = 0.0
        if lane_data is not None:
            lane_steering = self.lane_detector.get_steering_suggestion(
                lane_data,
                640  # image width
            )

        # 2. Obstacle avoidance
        obstacle_steering = self.obstacle_avoidance.calculate_avoidance_steering(
            current_pos,
            current_heading,
            lidar_data
        )

        # 3. Check collision risk and adjust speed
        collision_risk, min_distance = self.obstacle_avoidance.check_collision_risk(lidar_data)
        safe_speed = self.obstacle_avoidance.calculate_safe_speed(
            lidar_data,
            self.current_speed,
            self.target_speed
        )

        # Combine steering commands (weighted)
        if collision_risk:
            # Prioritize obstacle avoidance
            steering = obstacle_steering
            throttle = 0.0  # Brake
        else:
            # Combine lane following and obstacle avoidance
            steering = 0.7 * lane_steering + 0.3 * obstacle_steering
            throttle = safe_speed / 10.0  # Normalize to [-1, 1]

        # Limit commands
        steering = np.clip(steering, -0.5, 0.5)
        throttle = np.clip(throttle, -1.0, 1.0)

        return throttle, steering

    def _manual_control(self):
        """Manual control using keyboard"""
        throttle = 0.0
        steering = 0.0

        keys = p.getKeyboardEvents()

        # Arrow keys
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            throttle = 0.5
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            throttle = -0.5
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            steering = 0.3
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            steering = -0.3

        return throttle, steering

    def _handle_keyboard(self):
        """Handle keyboard input for simulator control"""
        keys = p.getKeyboardEvents()

        # Space: Toggle autonomous mode
        if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
            self.autonomous_mode = not self.autonomous_mode
            mode = "AUTONOMOUS" if self.autonomous_mode else "MANUAL"
            print(f"Switched to {mode} mode")

        # C: Toggle camera view
        if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
            self.show_camera = not self.show_camera
            print(f"Camera view: {'ON' if self.show_camera else 'OFF'}")

        # L: Toggle LIDAR visualization
        if ord('l') in keys and keys[ord('l')] & p.KEY_WAS_TRIGGERED:
            self.show_lidar = not self.show_lidar
            print(f"LIDAR visualization: {'ON' if self.show_lidar else 'OFF'}")

        # R: Reset simulation
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            print("Resetting simulation...")
            self.reset()

        # Q or ESC: Quit
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            self.running = False
        if p.B3G_ESCAPE in keys:
            self.running = False

    def _update_visualization(self, camera_rgb, lane_data, lidar_data, fused_state):
        """Update visualization windows"""
        if not self.gui:
            return

        # Display camera with lane detection
        if self.show_camera and camera_rgb is not None:
            display_image = camera_rgb.copy()

            if lane_data is not None:
                display_image = self.lane_detector.draw_lanes(display_image, lane_data)

            # Add sensor fusion info
            uncertainty = self.sensor_fusion.get_uncertainty()
            cv2.putText(
                display_image,
                f"Position Uncertainty: {uncertainty:.2f}m",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Add mode indicator
            mode_text = "AUTO" if self.autonomous_mode else "MANUAL"
            cv2.putText(
                display_image,
                f"Mode: {mode_text}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if self.autonomous_mode else (0, 0, 255),
                2
            )

            # Show position
            pos = fused_state['position']
            cv2.putText(
                display_image,
                f"Pos: ({pos[0]:.1f}, {pos[1]:.1f})",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Self-Driving Car Camera", display_image)
            cv2.waitKey(1)

        # Display LIDAR visualization
        if self.show_lidar and lidar_data:
            lidar_image = self._create_lidar_visualization(lidar_data)
            cv2.imshow("LIDAR Scan", lidar_image)
            cv2.waitKey(1)

    def _create_lidar_visualization(self, lidar_data):
        """Create LIDAR visualization image"""
        size = 500
        image = np.zeros((size, size, 3), dtype=np.uint8)
        center = size // 2
        scale = 5  # pixels per meter

        # Draw grid
        for i in range(0, size, 50):
            cv2.line(image, (i, 0), (i, size), (30, 30, 30), 1)
            cv2.line(image, (0, i), (size, i), (30, 30, 30), 1)

        # Draw car
        cv2.circle(image, (center, center), 10, (0, 255, 0), -1)

        # Draw LIDAR points
        for data in lidar_data:
            if data['object_id'] >= 0:
                angle = data['angle']
                distance = data['distance']

                x = int(center + distance * np.cos(angle) * scale)
                y = int(center + distance * np.sin(angle) * scale)

                if 0 <= x < size and 0 <= y < size:
                    color = (0, 0, 255) if distance < 5 else (255, 255, 0)
                    cv2.circle(image, (x, y), 2, color, -1)

        return image

    def reset(self):
        """Reset the simulation"""
        self.car.reset()
        self.sensor_fusion.reset()
        print("Simulation reset!")

    def cleanup(self):
        """Cleanup and close simulation"""
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        self.world.close()
        print("Simulation closed!")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Self-Driving Car Simulator")
    print("PyBullet-based autonomous vehicle simulation")
    print("=" * 60)
    print()

    # Create and run simulator
    simulator = SelfDrivingCarSimulator(gui=True)
    simulator.run()


if __name__ == "__main__":
    main()
