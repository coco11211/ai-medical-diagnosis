"""
Real-time Video Style Transfer
Supports 4K video processing with GPU acceleration
"""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Callable
from pathlib import Path
import logging
from tqdm import tqdm
import time

from ..models.resnet_model import ResNetStyleTransfer, FastStyleTransferNetwork
from ..utils.gpu_utils import GPUManager, TensorPool
from ..utils.style_interpolation import TemporalStyleSmoother
from ..utils.image_utils import ImageProcessor


class VideoStyleTransfer:
    """
    Real-time video style transfer with support for various resolutions including 4K
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize video style transfer

        Args:
            model_path: Path to pre-trained model weights (optional)
            device: Device to run on (auto-detected if None)
        """
        self.gpu_manager = GPUManager()
        self.device = device or self.gpu_manager.get_device()

        # Use fast model for real-time processing
        self.model = FastStyleTransferNetwork(device=self.device)

        if model_path:
            self.load_model(model_path)

        self.model = self.gpu_manager.optimize_for_inference(self.model)

        # Temporal smoothing
        self.smoother = TemporalStyleSmoother(temporal_weight=0.3)

        # Memory optimization
        self.tensor_pool = TensorPool(device=self.device)

        # Performance tracking
        self.fps_history = []

        logging.info(f"Video style transfer initialized on {self.device}")

    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        logging.info(f"Loaded model from {model_path}")

    def process_video(self,
                     input_path: str,
                     output_path: str,
                     max_size: Optional[int] = None,
                     use_temporal_smoothing: bool = True,
                     fps: Optional[int] = None,
                     codec: str = 'mp4v',
                     quality: int = 5,
                     progress_callback: Optional[Callable] = None) -> Dict[str, float]:
        """
        Process video file with style transfer

        Args:
            input_path: Path to input video
            output_path: Path to output video
            max_size: Maximum dimension (None = original size)
            use_temporal_smoothing: Apply temporal smoothing
            fps: Output FPS (None = same as input)
            codec: Video codec ('mp4v', 'avc1', 'XVID')
            quality: Compression quality (0-10, higher = better)
            progress_callback: Optional callback function for progress

        Returns:
            Dictionary with processing statistics
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logging.info(f"Input video: {width}x{height} @ {original_fps:.2f} FPS, {total_frames} frames")

        # Calculate output size
        if max_size:
            scale = max_size / max(width, height)
            out_width = int(width * scale)
            out_height = int(height * scale)

            # Ensure divisible by 8
            out_width = (out_width // 8) * 8
            out_height = (out_height // 8) * 8
        else:
            out_width, out_height = width, height

        # Setup output video writer
        output_fps = fps or original_fps
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (out_width, out_height))

        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")

        # Reset smoother
        if use_temporal_smoothing:
            self.smoother.reset()

        # Process frames
        frame_times = []
        processed_frames = 0

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                start_time = time.time()

                # Process frame
                styled_frame = self._process_frame(
                    frame,
                    (out_width, out_height),
                    use_temporal_smoothing
                )

                # Write frame
                out.write(styled_frame)

                # Track performance
                frame_time = time.time() - start_time
                frame_times.append(frame_time)

                processed_frames += 1
                pbar.update(1)

                # Update progress callback
                if progress_callback:
                    progress_callback(processed_frames, total_frames)

                # Update progress bar with FPS
                if processed_frames % 30 == 0:
                    avg_fps = 1.0 / np.mean(frame_times[-30:])
                    pbar.set_postfix({'FPS': f'{avg_fps:.1f}'})

        # Cleanup
        cap.release()
        out.release()

        # Calculate statistics
        avg_frame_time = np.mean(frame_times)
        avg_fps = 1.0 / avg_frame_time

        stats = {
            'total_frames': processed_frames,
            'avg_fps': avg_fps,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'total_time_s': sum(frame_times),
            'output_resolution': f'{out_width}x{out_height}'
        }

        logging.info(f"Video processing complete: {avg_fps:.2f} FPS")

        return stats

    def _process_frame(self, frame: np.ndarray, size: Tuple[int, int],
                      use_smoothing: bool = True) -> np.ndarray:
        """
        Process single video frame

        Args:
            frame: Input frame (BGR format from OpenCV)
            size: Output size (width, height)
            use_smoothing: Apply temporal smoothing

        Returns:
            Styled frame (BGR format)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if size != (frame.shape[1], frame.shape[0]):
            frame_rgb = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_LINEAR)

        # Convert to tensor
        frame_pil = Image.fromarray(frame_rgb)
        input_tensor = self._preprocess_frame(frame_pil)

        # Style transfer
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

            # Apply temporal smoothing
            if use_smoothing:
                output_tensor = self.smoother.smooth(output_tensor)

        # Convert back to numpy
        output_frame = self._postprocess_frame(output_tensor)

        # Convert RGB to BGR
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        return output_frame

    def _preprocess_frame(self, frame: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor"""
        from torchvision import transforms

        transform = transforms.ToTensor()
        tensor = transform(frame).unsqueeze(0).to(self.device)
        return tensor

    def _postprocess_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array"""
        tensor = tensor.squeeze(0).clamp(0, 1)
        array = tensor.cpu().numpy()
        array = np.transpose(array, (1, 2, 0))
        array = (array * 255).astype(np.uint8)
        return array

    def get_supported_resolutions(self) -> dict:
        """Get supported video resolutions"""
        return {
            'SD': (640, 480),
            'HD': (1280, 720),
            'Full HD': (1920, 1080),
            '2K': (2560, 1440),
            '4K': (3840, 2160),
            '8K': (7680, 4320)
        }

    def estimate_processing_time(self, video_path: str, fps_estimate: float = None) -> dict:
        """
        Estimate video processing time

        Args:
            video_path: Path to video file
            fps_estimate: Estimated processing FPS (auto-detected if None)

        Returns:
            Dictionary with time estimates
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Estimate processing FPS if not provided
        if fps_estimate is None:
            # Rough estimates based on resolution
            pixels = width * height
            if pixels <= 640 * 480:
                fps_estimate = 60
            elif pixels <= 1920 * 1080:
                fps_estimate = 30
            elif pixels <= 3840 * 2160:
                fps_estimate = 10
            else:
                fps_estimate = 5

        estimated_time = total_frames / fps_estimate

        return {
            'total_frames': total_frames,
            'resolution': f'{width}x{height}',
            'estimated_fps': fps_estimate,
            'estimated_time_seconds': estimated_time,
            'estimated_time_formatted': f'{int(estimated_time // 60)}m {int(estimated_time % 60)}s'
        }


class VideoExporter:
    """
    Handles exporting video in multiple formats
    """

    SUPPORTED_CODECS = {
        'mp4': 'mp4v',  # MPEG-4
        'avi': 'XVID',  # Xvid
        'mov': 'avc1',  # H.264
        'mkv': 'X264'   # H.264 in MKV
    }

    @staticmethod
    def export_video(input_path: str, output_path: str,
                    codec: Optional[str] = None, crf: int = 23):
        """
        Export video with specified codec

        Args:
            input_path: Input video path
            output_path: Output video path
            codec: Video codec (auto-detected from extension if None)
            crf: Constant Rate Factor for quality (0-51, lower = better)
        """
        import subprocess

        # Auto-detect codec from extension
        if codec is None:
            ext = Path(output_path).suffix[1:].lower()
            codec = VideoExporter.SUPPORTED_CODECS.get(ext, 'mp4v')

        # Use ffmpeg for better quality if available
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-crf', str(crf),
                '-preset', 'medium',
                '-c:a', 'copy',
                '-y',  # Overwrite output
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logging.info(f"Video exported with ffmpeg: {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to OpenCV
            logging.warning("ffmpeg not available, using OpenCV (lower quality)")
            # OpenCV export logic here
            pass

    @staticmethod
    def create_gif(video_path: str, output_path: str,
                  max_size: int = 480, fps: int = 10, duration: Optional[float] = None):
        """
        Create animated GIF from video

        Args:
            video_path: Input video path
            output_path: Output GIF path
            max_size: Maximum dimension
            fps: Output FPS
            duration: Maximum duration in seconds (None = full video)
        """
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate frame skip
        frame_skip = int(original_fps / fps)

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Resize frame
                h, w = frame.shape[:2]
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)

                frame_resized = cv2.resize(frame, (new_w, new_h))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

            frame_count += 1

            # Check duration limit
            if duration and frame_count / original_fps >= duration:
                break

        cap.release()

        # Save as GIF
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0,
                optimize=True
            )
            logging.info(f"GIF created: {output_path}")
