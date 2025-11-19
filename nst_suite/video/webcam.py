"""Webcam style transfer for live video."""

import torch
import numpy as np
import cv2
from typing import Optional
import time

from ..core.transfer import FastStyleTransfer, MultiStyleFastTransfer
from ..utils.video import WebcamProcessor
from ..utils.cuda import optimize_cuda_memory


class WebcamStyleTransfer:
    """
    Real-time webcam style transfer.
    """

    def __init__(
        self,
        model_path: str,
        camera_id: int = 0,
        device: Optional[torch.device] = None,
        resize_factor: float = 0.75
    ):
        """
        Initialize webcam style transfer.

        Args:
            model_path: Path to pre-trained model
            camera_id: Webcam device ID
            device: PyTorch device
            resize_factor: Factor to resize frames for faster processing
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.resize_factor = resize_factor

        # Initialize style transfer
        self.transfer = FastStyleTransfer(model_path, device=self.device)

        # Use half precision on CUDA for speed
        if torch.cuda.is_available():
            self.transfer.network = self.transfer.network.half()
            optimize_cuda_memory()

        # Initialize webcam
        self.webcam = WebcamProcessor(camera_id)

        # Warm up
        self._warmup()

    def _warmup(self):
        """Warm up the model."""
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        if torch.cuda.is_available():
            dummy_input = dummy_input.half()

        with torch.no_grad():
            for _ in range(3):
                _ = self.transfer.network(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single webcam frame.

        Args:
            frame: Input frame (H, W, C) in [0, 255]

        Returns:
            Stylized frame
        """
        original_size = frame.shape[:2]

        # Resize for faster processing
        if self.resize_factor != 1.0:
            new_h = int(frame.shape[0] * self.resize_factor)
            new_w = int(frame.shape[1] * self.resize_factor)
            frame = cv2.resize(frame, (new_w, new_h))

        # Preprocess
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        if torch.cuda.is_available():
            tensor = tensor.half()

        # Apply style transfer
        with torch.no_grad():
            stylized_tensor = self.transfer.network(tensor)

        # Postprocess
        stylized_frame = stylized_tensor[0].cpu().float().permute(1, 2, 0).numpy()
        stylized_frame = (stylized_frame * 255).clip(0, 255).astype(np.uint8)

        # Resize back
        if self.resize_factor != 1.0:
            stylized_frame = cv2.resize(
                stylized_frame,
                (original_size[1], original_size[0])
            )

        return stylized_frame

    def run(
        self,
        window_name: str = 'Webcam Style Transfer',
        target_fps: int = 30,
        save_output: Optional[str] = None,
        show_fps: bool = True
    ):
        """
        Run webcam style transfer.

        Args:
            window_name: Name for display window
            target_fps: Target frames per second
            save_output: Optional path to save output video
            show_fps: Whether to display FPS on screen
        """
        self.webcam.start()

        # Setup video writer if saving
        video_writer = None
        if save_output is not None:
            first_frame = self.webcam.read_frame()
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    save_output, fourcc, target_fps, (w, h)
                )

        frame_time = 1.0 / target_fps
        fps_history = []

        try:
            print(f"Starting webcam style transfer. Press 'q' to quit.")
            print(f"Target FPS: {target_fps}")

            while True:
                start_time = time.time()

                # Read frame
                frame = self.webcam.read_frame()
                if frame is None:
                    break

                # Process frame
                stylized_frame = self.process_frame(frame)

                # Add FPS overlay
                if show_fps:
                    current_fps = 1.0 / (time.time() - start_time + 1e-6)
                    fps_history.append(current_fps)
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                    avg_fps = sum(fps_history) / len(fps_history)

                    cv2.putText(
                        stylized_frame,
                        f'FPS: {avg_fps:.1f}',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )

                # Display frame
                display_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, display_frame)

                # Save frame if requested
                if video_writer is not None:
                    video_writer.write(display_frame)

                # FPS limiting
                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.webcam.stop()
            cv2.destroyAllWindows()
            if video_writer is not None:
                video_writer.release()

            if fps_history:
                avg_fps = sum(fps_history) / len(fps_history)
                print(f"\nAverage FPS: {avg_fps:.2f}")


class MultiStyleWebcam:
    """
    Webcam with multiple styles and interactive style switching.
    """

    def __init__(
        self,
        model_path: str,
        num_styles: int,
        camera_id: int = 0,
        device: Optional[torch.device] = None,
        resize_factor: float = 0.75
    ):
        """
        Initialize multi-style webcam.

        Args:
            model_path: Path to multi-style model
            num_styles: Number of styles
            camera_id: Webcam device ID
            device: PyTorch device
            resize_factor: Resize factor for processing
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.num_styles = num_styles
        self.resize_factor = resize_factor

        # Initialize multi-style transfer
        self.transfer = MultiStyleFastTransfer(
            num_styles=num_styles,
            model_path=model_path,
            device=self.device
        )

        if torch.cuda.is_available():
            self.transfer.network = self.transfer.network.half()
            optimize_cuda_memory()

        # Initialize webcam
        self.webcam = WebcamProcessor(camera_id)

        # Current style weights
        self.style_weights = np.zeros(num_styles)
        self.style_weights[0] = 1.0
        self.current_style_idx = 0

    def process_frame(
        self,
        frame: np.ndarray,
        style_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process frame with current style.

        Args:
            frame: Input frame
            style_weights: Optional style weights

        Returns:
            Stylized frame
        """
        if style_weights is None:
            style_weights = self.style_weights

        original_size = frame.shape[:2]

        # Resize
        if self.resize_factor != 1.0:
            new_h = int(frame.shape[0] * self.resize_factor)
            new_w = int(frame.shape[1] * self.resize_factor)
            frame = cv2.resize(frame, (new_w, new_h))

        # Process with style transfer
        stylized_frame = self.transfer.transfer(frame, style_weights)

        # Resize back
        if self.resize_factor != 1.0:
            stylized_frame = cv2.resize(
                stylized_frame,
                (original_size[1], original_size[0])
            )

        return stylized_frame

    def run(
        self,
        window_name: str = 'Multi-Style Webcam',
        target_fps: int = 30
    ):
        """
        Run multi-style webcam with interactive controls.

        Args:
            window_name: Display window name
            target_fps: Target frames per second
        """
        self.webcam.start()

        frame_time = 1.0 / target_fps

        try:
            print(f"Starting multi-style webcam. Press 'q' to quit.")
            print(f"Press 0-{self.num_styles-1} to switch styles")
            print(f"Press 's' to toggle smooth interpolation")

            interpolate = False

            while True:
                start_time = time.time()

                # Read frame
                frame = self.webcam.read_frame()
                if frame is None:
                    break

                # Process frame
                stylized_frame = self.process_frame(frame)

                # Add style indicator
                style_text = f'Style: {self.current_style_idx}'
                if interpolate:
                    style_text += ' (Interpolating)'

                cv2.putText(
                    stylized_frame,
                    style_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Display
                display_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    interpolate = not interpolate
                elif ord('0') <= key <= ord('9'):
                    new_style_idx = key - ord('0')
                    if new_style_idx < self.num_styles:
                        if interpolate:
                            # Smooth transition
                            target_weights = np.zeros(self.num_styles)
                            target_weights[new_style_idx] = 1.0
                            for alpha in np.linspace(0, 1, 10):
                                self.style_weights = (
                                    (1 - alpha) * self.style_weights +
                                    alpha * target_weights
                                )
                        else:
                            # Instant switch
                            self.style_weights = np.zeros(self.num_styles)
                            self.style_weights[new_style_idx] = 1.0

                        self.current_style_idx = new_style_idx

                # FPS limiting
                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

        finally:
            self.webcam.stop()
            cv2.destroyAllWindows()


def main():
    """Main function for webcam style transfer CLI."""
    import argparse

    parser = argparse.ArgumentParser(description='Webcam style transfer')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to style transfer model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--resize-factor', type=float, default=0.75,
                        help='Resize factor for faster processing')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS')
    parser.add_argument('--save', type=str, default=None,
                        help='Save output to video file')

    args = parser.parse_args()

    webcam = WebcamStyleTransfer(
        model_path=args.model,
        camera_id=args.camera,
        resize_factor=args.resize_factor
    )

    webcam.run(
        target_fps=args.fps,
        save_output=args.save
    )


if __name__ == '__main__':
    main()
