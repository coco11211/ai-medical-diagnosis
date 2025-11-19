"""
Vision System Module
Provides camera integration, calibration, and coordinate transformation
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraCalibration:
    """Camera calibration parameters"""
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    dist_coeffs: np.ndarray  # Distortion coefficients
    rvec: Optional[np.ndarray] = None  # Rotation vector (camera to world)
    tvec: Optional[np.ndarray] = None  # Translation vector (camera to world)


class VisionSystem:
    """
    Vision system for robot arm with camera calibration and coordinate transformation
    """

    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1920, 1080),
        calibration: Optional[CameraCalibration] = None
    ):
        """
        Initialize vision system

        Args:
            camera_id: Camera device ID (0 for default webcam)
            resolution: Camera resolution (width, height)
            calibration: Camera calibration parameters
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.calibration = calibration
        self.camera = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """
        Initialize camera

        Returns:
            True if successful
        """
        try:
            self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
            if not self.camera.isOpened():
                print(f"Failed to open camera {self.camera_id}")
                return False

            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Verify resolution
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera initialized: {int(actual_width)}x{int(actual_height)}")

            self.is_initialized = True
            return True

        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from camera

        Returns:
            Image as numpy array (BGR format) or None if failed
        """
        if not self.is_initialized:
            print("Camera not initialized")
            return None

        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame")
            return None

        return frame

    def calibrate_camera(
        self,
        num_images: int = 20,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025
    ) -> bool:
        """
        Calibrate camera using checkerboard pattern

        Args:
            num_images: Number of calibration images to capture
            checkerboard_size: Inner corners (cols, rows)
            square_size: Size of checkerboard square in meters

        Returns:
            True if calibration successful
        """
        if not self.is_initialized:
            print("Camera not initialized")
            return False

        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        obj_points = []  # 3D points in real world
        img_points = []  # 2D points in image

        print(f"Capturing {num_images} calibration images...")
        print("Press SPACE to capture, ESC to cancel")

        captured = 0
        while captured < num_images:
            frame = self.capture_frame()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            # Draw corners
            display = frame.copy()
            if ret:
                cv2.drawChessboardCorners(display, checkerboard_size, corners, ret)
                cv2.putText(display, f"Captured: {captured}/{num_images} - Press SPACE",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Checkerboard not found", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Camera Calibration', display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return False
            elif key == 32 and ret:  # SPACE
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                obj_points.append(objp)
                img_points.append(corners_refined)
                captured += 1
                print(f"Captured {captured}/{num_images}")

        cv2.destroyAllWindows()

        # Calibrate camera
        print("Performing calibration...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )

        if ret:
            self.calibration = CameraCalibration(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs
            )
            print("Calibration successful!")
            print(f"Camera Matrix:\n{camera_matrix}")
            print(f"Distortion Coefficients:\n{dist_coeffs}")
            return True
        else:
            print("Calibration failed")
            return False

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort image using calibration parameters

        Args:
            image: Input image

        Returns:
            Undistorted image
        """
        if self.calibration is None:
            return image

        return cv2.undistort(
            image,
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs
        )

    def pixel_to_camera_coordinates(
        self,
        pixel_coords: np.ndarray,
        depth: float
    ) -> np.ndarray:
        """
        Convert pixel coordinates to camera coordinates

        Args:
            pixel_coords: Pixel coordinates [u, v]
            depth: Depth in meters (Z coordinate)

        Returns:
            3D coordinates in camera frame [X, Y, Z]
        """
        if self.calibration is None:
            raise ValueError("Camera not calibrated")

        fx = self.calibration.camera_matrix[0, 0]
        fy = self.calibration.camera_matrix[1, 1]
        cx = self.calibration.camera_matrix[0, 2]
        cy = self.calibration.camera_matrix[1, 2]

        u, v = pixel_coords
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z])

    def camera_to_world_coordinates(
        self,
        camera_coords: np.ndarray
    ) -> np.ndarray:
        """
        Transform camera coordinates to world (robot base) coordinates

        Args:
            camera_coords: 3D coordinates in camera frame

        Returns:
            3D coordinates in world frame
        """
        if self.calibration is None or self.calibration.rvec is None:
            raise ValueError("Camera-to-world transformation not calibrated")

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(self.calibration.rvec)

        # Transform: world = R * camera + t
        world_coords = R @ camera_coords + self.calibration.tvec.flatten()

        return world_coords

    def calibrate_hand_eye(
        self,
        robot_poses: List[np.ndarray],
        marker_size: float = 0.1,
        marker_id: int = 0
    ) -> bool:
        """
        Perform hand-eye calibration (camera to robot base transformation)

        Args:
            robot_poses: List of robot end-effector poses (4x4 transformation matrices)
            marker_size: ArUco marker size in meters
            marker_id: ArUco marker ID

        Returns:
            True if calibration successful
        """
        if not self.is_initialized or self.calibration is None:
            print("Camera not initialized or not calibrated")
            return False

        # ArUco marker detection
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        camera_poses = []

        print(f"Capturing {len(robot_poses)} poses for hand-eye calibration...")

        for i, robot_pose in enumerate(robot_poses):
            print(f"Move robot to pose {i+1}/{len(robot_poses)} and press SPACE")

            while True:
                frame = self.capture_frame()
                if frame is None:
                    continue

                # Detect ArUco markers
                corners, ids, _ = detector.detectMarkers(frame)

                display = frame.copy()
                if ids is not None and marker_id in ids:
                    # Draw detected markers
                    cv2.aruco.drawDetectedMarkers(display, corners, ids)

                    # Estimate pose
                    idx = np.where(ids == marker_id)[0][0]
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[idx],
                        marker_size,
                        self.calibration.camera_matrix,
                        self.calibration.dist_coeffs
                    )

                    # Draw axis
                    cv2.drawFrameAxes(
                        display,
                        self.calibration.camera_matrix,
                        self.calibration.dist_coeffs,
                        rvec, tvec, marker_size * 0.5
                    )

                    cv2.putText(display, f"Marker detected - Press SPACE ({i+1}/{len(robot_poses)})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('Hand-Eye Calibration', display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # SPACE
                        # Store camera pose
                        R, _ = cv2.Rodrigues(rvec)
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = tvec.flatten()
                        camera_poses.append(T)
                        break
                    elif key == 27:  # ESC
                        cv2.destroyAllWindows()
                        return False
                else:
                    cv2.putText(display, "Marker not found", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Hand-Eye Calibration', display)
                    cv2.waitKey(1)

        cv2.destroyAllWindows()

        # TODO: Implement actual hand-eye calibration algorithm
        # For now, store identity transformation
        self.calibration.rvec = np.zeros(3)
        self.calibration.tvec = np.zeros((3, 1))

        print("Hand-eye calibration completed")
        return True

    def detect_objects(
        self,
        frame: np.ndarray,
        method: str = 'contour'
    ) -> List[Dict]:
        """
        Detect objects in frame

        Args:
            frame: Input image
            method: Detection method ('contour', 'blob', 'template')

        Returns:
            List of detected objects with properties
        """
        if method == 'contour':
            return self._detect_by_contour(frame)
        elif method == 'blob':
            return self._detect_by_blob(frame)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _detect_by_contour(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using contour detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # Filter small contours
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Get centroid
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2

            objects.append({
                'centroid': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
                'contour': cnt
            })

        return objects

    def _detect_by_blob(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using blob detection"""
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 100
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(frame)

        objects = []
        for kp in keypoints:
            objects.append({
                'centroid': (int(kp.pt[0]), int(kp.pt[1])),
                'radius': kp.size,
                'keypoint': kp
            })

        return objects

    def save_calibration(self, filepath: str) -> bool:
        """Save camera calibration to file"""
        if self.calibration is None:
            return False

        try:
            np.savez(
                filepath,
                camera_matrix=self.calibration.camera_matrix,
                dist_coeffs=self.calibration.dist_coeffs,
                rvec=self.calibration.rvec,
                tvec=self.calibration.tvec
            )
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def load_calibration(self, filepath: str) -> bool:
        """Load camera calibration from file"""
        try:
            data = np.load(filepath)
            self.calibration = CameraCalibration(
                camera_matrix=data['camera_matrix'],
                dist_coeffs=data['dist_coeffs'],
                rvec=data['rvec'] if 'rvec' in data else None,
                tvec=data['tvec'] if 'tvec' in data else None
            )
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

    def release(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            cv2.destroyAllWindows()
            self.is_initialized = False
