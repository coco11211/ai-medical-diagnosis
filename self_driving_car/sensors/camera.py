"""
Camera sensor for the self-driving car
"""
import pybullet as p
import numpy as np


class Camera:
    """Camera sensor for visual perception"""

    def __init__(self, width=640, height=480, fov=60):
        """
        Initialize camera sensor

        Args:
            width: Image width
            height: Image height
            fov: Field of view in degrees
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.aspect = width / height
        self.near = 0.1
        self.far = 100

    def capture(self, car_position, car_orientation):
        """
        Capture an image from the car's perspective

        Args:
            car_position: Car position [x, y, z]
            car_orientation: Car orientation quaternion

        Returns:
            RGB image as numpy array
        """
        # Convert quaternion to euler
        euler = p.getEulerFromQuaternion(car_orientation)
        yaw = euler[2]

        # Camera position (mounted on top of car)
        cam_height = 0.3
        cam_pos = [
            car_position[0],
            car_position[1],
            car_position[2] + cam_height
        ]

        # Target position (looking forward)
        target_distance = 10
        target_pos = [
            car_position[0] + target_distance * np.cos(yaw),
            car_position[1] + target_distance * np.sin(yaw),
            car_position[2] + cam_height
        ]

        # Up vector
        up_vector = [0, 0, 1]

        # View matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector
        )

        # Projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far
        )

        # Capture image
        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert to numpy array
        rgb_array = np.array(rgb, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel

        depth_array = np.array(depth, dtype=np.float32)

        return rgb_array, depth_array, seg

    def get_specs(self):
        """Get camera specifications"""
        return {
            'width': self.width,
            'height': self.height,
            'fov': self.fov,
            'aspect': self.aspect
        }
