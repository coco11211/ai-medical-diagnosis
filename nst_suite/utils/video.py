"""Video processing utilities."""

import cv2
import numpy as np
from typing import Optional, Callable, Generator
import os
from pathlib import Path


class VideoProcessor:
    """
    Video processing utilities for style transfer.
    """

    def __init__(self):
        """Initialize video processor."""
        self.supported_formats = {
            '.mp4': 'mp4v',
            '.avi': 'XVID',
            '.mov': 'mp4v',
            '.mkv': 'X264',
        }

    def read_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Read video frames.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to read

        Yields:
            Video frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            cap.release()

    def write_video(
        self,
        frames: list,
        output_path: str,
        fps: float = 30.0,
        fourcc: Optional[str] = None
    ):
        """
        Write frames to video file.

        Args:
            frames: List of frames as numpy arrays (RGB)
            output_path: Path to output video
            fps: Frames per second
            fourcc: Four-character code for video codec
        """
        if len(frames) == 0:
            raise ValueError("No frames to write")

        # Get frame dimensions
        h, w = frames[0].shape[:2]

        # Determine codec
        if fourcc is None:
            ext = os.path.splitext(output_path)[1].lower()
            fourcc = self.supported_formats.get(ext, 'mp4v')

        # Create video writer
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        out = cv2.VideoWriter(output_path, fourcc_code, fps, (w, h))

        try:
            for frame in frames:
                # Convert RGB to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        finally:
            out.release()

    def process_video(
        self,
        video_path: str,
        output_path: str,
        process_fn: Callable[[np.ndarray], np.ndarray],
        show_progress: bool = True,
        max_frames: Optional[int] = None
    ):
        """
        Process video with a custom function.

        Args:
            video_path: Path to input video
            output_path: Path to output video
            process_fn: Function to process each frame
            show_progress: Whether to show progress bar
            max_frames: Maximum number of frames to process
        """
        from tqdm import tqdm

        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        # Process frames
        processed_frames = []

        frame_generator = self.read_video(video_path, max_frames)
        if show_progress:
            frame_generator = tqdm(
                frame_generator,
                total=total_frames,
                desc="Processing video"
            )

        for frame in frame_generator:
            processed_frame = process_fn(frame)
            processed_frames.append(processed_frame)

        # Write output video
        self.write_video(processed_frames, output_path, fps)

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video information.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0,
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        }

        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']

        cap.release()
        return info

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        interval: int = 1,
        max_frames: Optional[int] = None
    ):
        """
        Extract frames from video.

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract
        """
        os.makedirs(output_dir, exist_ok=True)

        frame_count = 0
        saved_count = 0

        for frame in self.read_video(video_path):
            if frame_count % interval == 0:
                output_path = os.path.join(
                    output_dir,
                    f'frame_{saved_count:06d}.png'
                )
                # Convert RGB to BGR for saving
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, frame_bgr)

                saved_count += 1
                if max_frames is not None and saved_count >= max_frames:
                    break

            frame_count += 1

    def create_video_from_frames(
        self,
        frame_dir: str,
        output_path: str,
        fps: float = 30.0,
        pattern: str = '*.png'
    ):
        """
        Create video from image frames.

        Args:
            frame_dir: Directory containing frames
            output_path: Path to output video
            fps: Frames per second
            pattern: Glob pattern for frame files
        """
        from glob import glob

        # Get sorted list of frames
        frame_files = sorted(glob(os.path.join(frame_dir, pattern)))

        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {frame_dir}")

        # Load frames
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        # Write video
        self.write_video(frames, output_path, fps)


class WebcamProcessor:
    """
    Webcam processing for real-time style transfer.
    """

    def __init__(self, camera_id: int = 0):
        """
        Initialize webcam processor.

        Args:
            camera_id: Camera device ID
        """
        self.camera_id = camera_id
        self.cap = None

    def start(self):
        """Start webcam capture."""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {self.camera_id}")

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def stop(self):
        """Stop webcam capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from webcam.

        Returns:
            Frame as numpy array (RGB) or None if no frame available
        """
        if self.cap is None:
            raise RuntimeError("Webcam not started. Call start() first.")

        ret, frame = self.cap.read()

        if not ret:
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def process_stream(
        self,
        process_fn: Callable[[np.ndarray], np.ndarray],
        window_name: str = 'Style Transfer',
        target_fps: Optional[int] = 30,
        save_output: Optional[str] = None
    ):
        """
        Process webcam stream in real-time.

        Args:
            process_fn: Function to process each frame
            window_name: Name for display window
            target_fps: Target frames per second
            save_output: Optional path to save output video
        """
        import time

        self.start()

        # Setup video writer if saving
        video_writer = None
        if save_output is not None:
            first_frame = self.read_frame()
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    save_output, fourcc, target_fps or 30, (w, h)
                )

        # Calculate frame time
        frame_time = 1.0 / target_fps if target_fps else 0

        try:
            print(f"Starting webcam stream. Press 'q' to quit.")

            while True:
                start_time = time.time()

                # Read frame
                frame = self.read_frame()
                if frame is None:
                    break

                # Process frame
                processed_frame = process_fn(frame)

                # Display frame
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, display_frame)

                # Save frame if requested
                if video_writer is not None:
                    video_writer.write(display_frame)

                # FPS limiting
                elapsed = time.time() - start_time
                if target_fps and elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.stop()
            cv2.destroyAllWindows()
            if video_writer is not None:
                video_writer.release()

    def get_camera_info(self) -> dict:
        """
        Get camera information.

        Returns:
            Dictionary with camera properties
        """
        if self.cap is None:
            self.start()

        info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'backend': self.cap.getBackendName(),
        }

        return info

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
