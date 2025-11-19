"""Video style transfer processors."""

import torch
import numpy as np
from typing import Optional
from pathlib import Path
import cv2
from tqdm import tqdm

from ..core.transfer import FastStyleTransfer
from ..utils.video import VideoProcessor
from ..utils.cuda import optimize_cuda_memory


class VideoStyleTransfer:
    """
    Video style transfer with support for various formats and resolutions.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        use_half_precision: bool = False
    ):
        """
        Initialize video style transfer.

        Args:
            model_path: Path to pre-trained style transfer model
            device: PyTorch device
            use_half_precision: Use FP16 for faster processing
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize style transfer
        self.transfer = FastStyleTransfer(model_path, device=self.device)
        self.use_half_precision = use_half_precision

        if use_half_precision and torch.cuda.is_available():
            self.transfer.network = self.transfer.network.half()

        # Optimize CUDA memory
        if torch.cuda.is_available():
            optimize_cuda_memory()

        self.video_processor = VideoProcessor()

    def process_video(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 4,
        max_frames: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Process video file with style transfer.

        Args:
            input_path: Path to input video
            output_path: Path to output video
            batch_size: Number of frames to process in parallel
            max_frames: Maximum number of frames to process
            show_progress: Whether to show progress bar
        """
        # Get video info
        video_info = self.video_processor.get_video_info(input_path)
        total_frames = video_info['frame_count']

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        print(f"Processing video: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f} fps")
        print(f"Total frames: {total_frames}")

        # Process frames in batches
        processed_frames = []
        frame_buffer = []

        frame_generator = self.video_processor.read_video(input_path, max_frames)
        if show_progress:
            frame_generator = tqdm(
                frame_generator,
                total=total_frames,
                desc="Processing video"
            )

        for frame in frame_generator:
            frame_buffer.append(frame)

            # Process batch when buffer is full
            if len(frame_buffer) >= batch_size:
                stylized_batch = self._process_batch(frame_buffer)
                processed_frames.extend(stylized_batch)
                frame_buffer = []

        # Process remaining frames
        if len(frame_buffer) > 0:
            stylized_batch = self._process_batch(frame_buffer)
            processed_frames.extend(stylized_batch)

        # Write output video
        print(f"Writing output video to {output_path}")
        self.video_processor.write_video(
            processed_frames,
            output_path,
            fps=video_info['fps']
        )

        print("Video processing complete!")

    def _process_batch(self, frames: list) -> list:
        """
        Process a batch of frames.

        Args:
            frames: List of frames as numpy arrays

        Returns:
            List of stylized frames
        """
        # Preprocess frames
        frame_tensors = []
        for frame in frames:
            tensor = torch.from_numpy(frame).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            frame_tensors.append(tensor)

        # Batch tensors
        batch = torch.cat(frame_tensors, dim=0).to(self.device)

        if self.use_half_precision and torch.cuda.is_available():
            batch = batch.half()

        # Apply style transfer
        with torch.no_grad():
            stylized_batch = self.transfer.network(batch)

        # Convert back to numpy
        stylized_frames = []
        for i in range(stylized_batch.size(0)):
            stylized_tensor = stylized_batch[i].cpu().float()
            stylized_frame = stylized_tensor.permute(1, 2, 0).numpy()
            stylized_frame = (stylized_frame * 255).clip(0, 255).astype(np.uint8)
            stylized_frames.append(stylized_frame)

        return stylized_frames


class RealtimeVideoProcessor:
    """
    Real-time video style transfer with optimizations for low latency.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        target_fps: int = 30,
        resize_factor: float = 1.0
    ):
        """
        Initialize real-time processor.

        Args:
            model_path: Path to pre-trained model
            device: PyTorch device
            target_fps: Target frames per second
            resize_factor: Factor to resize frames (< 1 for faster processing)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.target_fps = target_fps
        self.resize_factor = resize_factor

        # Initialize style transfer
        self.transfer = FastStyleTransfer(model_path, device=self.device)

        # Use half precision on CUDA for speed
        if torch.cuda.is_available():
            self.transfer.network = self.transfer.network.half()
            optimize_cuda_memory()

        # Warm up the model
        self._warmup()

    def _warmup(self):
        """Warm up the model for consistent performance."""
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        if torch.cuda.is_available():
            dummy_input = dummy_input.half()

        with torch.no_grad():
            for _ in range(5):
                _ = self.transfer.network(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame in real-time.

        Args:
            frame: Input frame as numpy array (H, W, C) in [0, 255]

        Returns:
            Stylized frame
        """
        original_size = frame.shape[:2]

        # Resize if needed
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

        # Resize back if needed
        if self.resize_factor != 1.0:
            stylized_frame = cv2.resize(
                stylized_frame,
                (original_size[1], original_size[0])
            )

        return stylized_frame

    def process_video_file(
        self,
        input_path: str,
        output_path: str,
        show_preview: bool = False
    ):
        """
        Process video file in real-time mode.

        Args:
            input_path: Path to input video
            output_path: Path to output video
            show_preview: Whether to show preview window
        """
        import time

        video_processor = VideoProcessor()
        video_info = video_processor.get_video_info(input_path)

        cap = cv2.VideoCapture(input_path)
        fps = video_info['fps']
        w, h = video_info['width'], video_info['height']

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_count = 0
        total_time = 0

        try:
            print(f"Processing video in real-time mode...")
            print(f"Target FPS: {self.target_fps}")

            while True:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame
                stylized_frame = self.process_frame(frame_rgb)

                # Convert back to BGR
                stylized_bgr = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)

                # Write frame
                out.write(stylized_bgr)

                # Show preview if requested
                if show_preview:
                    cv2.imshow('Real-time Style Transfer', stylized_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Calculate FPS
                elapsed = time.time() - start_time
                total_time += elapsed
                frame_count += 1

                if frame_count % 30 == 0:
                    avg_fps = frame_count / total_time
                    print(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.2f}")

        finally:
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()

        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")


def main():
    """Main function for video processing CLI."""
    import argparse

    parser = argparse.ArgumentParser(description='Process video with style transfer')
    parser.add_argument('--input', type=str, required=True,
                        help='Input video path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output video path')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to style transfer model')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for processing')
    parser.add_argument('--realtime', action='store_true',
                        help='Use real-time processing mode')
    parser.add_argument('--resize-factor', type=float, default=1.0,
                        help='Resize factor for faster processing')
    parser.add_argument('--preview', action='store_true',
                        help='Show preview window')

    args = parser.parse_args()

    if args.realtime:
        processor = RealtimeVideoProcessor(
            model_path=args.model,
            resize_factor=args.resize_factor
        )
        processor.process_video_file(
            args.input,
            args.output,
            show_preview=args.preview
        )
    else:
        processor = VideoStyleTransfer(model_path=args.model)
        processor.process_video(
            args.input,
            args.output,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()
