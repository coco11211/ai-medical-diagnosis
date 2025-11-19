"""
Real-time Webcam Style Transfer
Live video processing with interactive controls
"""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional, Callable
import logging
import time
from collections import deque

from ..models.resnet_model import FastStyleTransferNetwork
from ..utils.gpu_utils import GPUManager
from ..utils.style_interpolation import TemporalStyleSmoother


class WebcamStyleTransfer:
    """
    Real-time webcam style transfer with interactive controls
    """

    def __init__(self, camera_id: int = 0, device: Optional[str] = None):
        """
        Initialize webcam style transfer

        Args:
            camera_id: Camera device ID (default: 0)
            device: Device to run on (auto-detected if None)
        """
        self.gpu_manager = GPUManager()
        self.device = device or self.gpu_manager.get_device()
        self.camera_id = camera_id

        # Initialize model
        self.model = FastStyleTransferNetwork(device=self.device)
        self.model = self.gpu_manager.optimize_for_inference(self.model)

        # Temporal smoothing
        self.smoother = TemporalStyleSmoother(temporal_weight=0.5)

        # Camera
        self.cap = None

        # FPS tracking
        self.fps_buffer = deque(maxlen=30)
        self.current_fps = 0

        # Settings
        self.use_smoothing = True
        self.show_fps = True
        self.mirror = True

        logging.info(f"Webcam style transfer initialized on {self.device}")

    def start(self, resolution: tuple = (1280, 720),
             window_name: str = "Neural Style Transfer - Webcam",
             save_output: bool = False,
             output_path: Optional[str] = None):
        """
        Start webcam processing

        Args:
            resolution: Camera resolution (width, height)
            window_name: OpenCV window name
            save_output: Whether to save output video
            output_path: Path for output video (if save_output=True)
        """
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        # Set resolution
        width, height = resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logging.info(f"Camera opened: {actual_width}x{actual_height}")

        # Setup video writer if saving
        out = None
        if save_output:
            if output_path is None:
                output_path = f"webcam_output_{int(time.time())}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (actual_width, actual_height))

        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Reset smoother
        self.smoother.reset()

        print("\n" + "="*60)
        print("Neural Style Transfer - Webcam Mode")
        print("="*60)
        print("Controls:")
        print("  Q or ESC  - Quit")
        print("  S         - Toggle temporal smoothing")
        print("  F         - Toggle FPS display")
        print("  M         - Toggle mirror mode")
        print("  SPACE     - Take screenshot")
        print("="*60 + "\n")

        frame_count = 0

        try:
            while True:
                start_time = time.time()

                # Capture frame
                ret, frame = self.cap.read()

                if not ret:
                    logging.error("Failed to capture frame")
                    break

                # Mirror if enabled
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                # Process frame
                styled_frame = self._process_frame(frame)

                # Calculate FPS
                frame_time = time.time() - start_time
                self.fps_buffer.append(1.0 / max(frame_time, 0.001))
                self.current_fps = np.mean(self.fps_buffer)

                # Add FPS overlay
                if self.show_fps:
                    self._add_fps_overlay(styled_frame)

                # Display
                cv2.imshow(window_name, styled_frame)

                # Save if recording
                if save_output and out:
                    out.write(styled_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s'):
                    self.use_smoothing = not self.use_smoothing
                    print(f"Temporal smoothing: {'ON' if self.use_smoothing else 'OFF'}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                elif key == ord('m'):
                    self.mirror = not self.mirror
                    print(f"Mirror mode: {'ON' if self.mirror else 'OFF'}")
                elif key == ord(' '):  # SPACE
                    screenshot_path = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(screenshot_path, styled_frame)
                    print(f"Screenshot saved: {screenshot_path}")

                frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

            print(f"\nProcessed {frame_count} frames")
            print(f"Average FPS: {self.current_fps:.2f}")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame

        Args:
            frame: Input frame (BGR)

        Returns:
            Styled frame (BGR)
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Convert to tensor
        from torchvision import transforms
        transform = transforms.ToTensor()
        input_tensor = transform(frame_pil).unsqueeze(0).to(self.device)

        # Style transfer
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

            # Apply smoothing
            if self.use_smoothing:
                output_tensor = self.smoother.smooth(output_tensor)

        # Convert back
        output_tensor = output_tensor.squeeze(0).clamp(0, 1)
        output_array = output_tensor.cpu().numpy()
        output_array = np.transpose(output_array, (1, 2, 0))
        output_array = (output_array * 255).astype(np.uint8)

        # Convert to BGR
        output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

        return output_bgr

    def _add_fps_overlay(self, frame: np.ndarray):
        """Add FPS overlay to frame"""
        fps_text = f"FPS: {self.current_fps:.1f}"

        # Add background rectangle
        (text_width, text_height), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (20 + text_width, 20 + text_height),
            (0, 0, 0),
            -1
        )

        # Add text
        cv2.putText(
            frame,
            fps_text,
            (15, 15 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Add device info
        device_text = f"Device: {self.device.upper()}"
        cv2.putText(
            frame,
            device_text,
            (15, 50 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    def stop(self):
        """Stop webcam processing"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def load_model(self, model_path: str):
        """Load pre-trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        logging.info(f"Model loaded from {model_path}")

    def get_available_cameras(self) -> list:
        """
        Get list of available cameras

        Returns:
            List of camera IDs
        """
        available = []

        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()

        return available

    def set_camera_settings(self, brightness: Optional[float] = None,
                           contrast: Optional[float] = None,
                           saturation: Optional[float] = None):
        """
        Adjust camera settings

        Args:
            brightness: Brightness value (0-1)
            contrast: Contrast value (0-1)
            saturation: Saturation value (0-1)
        """
        if not self.cap or not self.cap.isOpened():
            logging.warning("Camera not opened")
            return

        if brightness is not None:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        if contrast is not None:
            self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        if saturation is not None:
            self.cap.set(cv2.CAP_PROP_SATURATION, saturation)


class DualViewWebcam:
    """
    Side-by-side view of original and styled webcam feed
    """

    def __init__(self, camera_id: int = 0, device: Optional[str] = None):
        self.webcam = WebcamStyleTransfer(camera_id, device)

    def start(self, resolution: tuple = (1280, 720)):
        """Start dual view mode"""
        self.webcam.cap = cv2.VideoCapture(self.webcam.camera_id)

        if not self.webcam.cap.isOpened():
            raise RuntimeError(f"Cannot open camera")

        width, height = resolution
        self.webcam.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.webcam.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(self.webcam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.webcam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        window_name = "Neural Style Transfer - Dual View (Original | Styled)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "="*60)
        print("Dual View Mode - Original | Styled")
        print("="*60)
        print("Press Q or ESC to quit")
        print("="*60 + "\n")

        try:
            while True:
                ret, frame = self.webcam.cap.read()

                if not ret:
                    break

                # Mirror
                if self.webcam.mirror:
                    frame = cv2.flip(frame, 1)

                # Process
                styled = self.webcam._process_frame(frame)

                # Create side-by-side view
                combined = np.hstack([frame, styled])

                # Add divider line
                h, w = combined.shape[:2]
                cv2.line(combined, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

                # Add labels
                cv2.putText(combined, "ORIGINAL", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "STYLED", (w // 2 + 20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow(window_name, combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        finally:
            self.webcam.stop()
