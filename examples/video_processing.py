"""
Video Style Transfer Example

Demonstrates different methods for processing videos:
1. Standard batch processing
2. Real-time processing
3. 4K video support
"""

import os
from nst_suite.video.processor import VideoStyleTransfer, RealtimeVideoProcessor
from nst_suite.utils.video import VideoProcessor


def main():
    """Run video processing examples."""

    print("="*60)
    print("Video Style Transfer Example")
    print("="*60)

    # Setup paths
    input_video = "input.mp4"
    model_path = "./checkpoints/final_model.pth"
    output_dir = "./outputs"

    os.makedirs(output_dir, exist_ok=True)

    # Check if files exist
    if not os.path.exists(input_video):
        print(f"\nError: Input video not found at {input_video}")
        print("Please provide an input video.")
        return

    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train a model first:")
        print("  nst train style.jpg ./dataset")
        return

    # Get video information
    video_proc = VideoProcessor()
    video_info = video_proc.get_video_info(input_video)

    print(f"\nInput Video Information:")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Frames: {video_info['frame_count']}")
    print(f"  Duration: {video_info['duration']:.2f} seconds")

    # Example 1: Standard processing (best quality)
    print("\n" + "="*60)
    print("Example 1: Standard Batch Processing")
    print("="*60)

    processor = VideoStyleTransfer(
        model_path=model_path,
        use_half_precision=True
    )

    output_path = os.path.join(output_dir, "output_standard.mp4")

    print(f"Processing video...")
    print(f"Using batch processing for best quality")

    processor.process_video(
        input_path=input_video,
        output_path=output_path,
        batch_size=4,
        show_progress=True
    )

    print(f"Saved to: {output_path}")

    # Example 2: Real-time processing (faster)
    print("\n" + "="*60)
    print("Example 2: Real-time Processing")
    print("="*60)

    realtime_processor = RealtimeVideoProcessor(
        model_path=model_path,
        target_fps=30,
        resize_factor=0.75  # Reduce resolution for speed
    )

    output_path = os.path.join(output_dir, "output_realtime.mp4")

    print(f"Processing video in real-time mode...")
    print(f"Using resize factor 0.75 for better performance")

    realtime_processor.process_video_file(
        input_path=input_video,
        output_path=output_path,
        show_preview=False
    )

    print(f"Saved to: {output_path}")

    # Example 3: 4K video processing
    if video_info['width'] >= 3840 or video_info['height'] >= 2160:
        print("\n" + "="*60)
        print("Example 3: 4K Video Processing")
        print("="*60)

        print("Processing 4K video with optimizations...")

        processor_4k = VideoStyleTransfer(
            model_path=model_path,
            use_half_precision=True
        )

        output_path = os.path.join(output_dir, "output_4k.mp4")

        # Use smaller batch size for 4K
        processor_4k.process_video(
            input_path=input_video,
            output_path=output_path,
            batch_size=2,  # Smaller batch for 4K
            show_progress=True
        )

        print(f"Saved to: {output_path}")

    else:
        print("\nSkipping 4K example (input video is not 4K)")

    print("\n" + "="*60)
    print("Video Processing Complete!")
    print("="*60)
    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()
