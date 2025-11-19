"""
Video Processing Example for Neural Style Transfer
Demonstrates video processing features including 4K support
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def example_basic_video():
    """Example: Basic video processing"""
    print("="*60)
    print("Example 1: Basic Video Processing")
    print("="*60)

    print("\nCLI Command:")
    print("  python style_transfer_main.py video \\")
    print("    input.mp4 output.mp4 \\")
    print("    --model-path trained_model.pth")

    print("\nPython API:")
    print("  from neural_style_transfer import VideoStyleTransfer")
    print("  processor = VideoStyleTransfer(model_path='model.pth')")
    print("  stats = processor.process_video('input.mp4', 'output.mp4')")
    print("  print(f'Processed at {stats[\"avg_fps\"]:.2f} FPS')")


def example_4k_video():
    """Example: 4K video processing"""
    print("\n" + "="*60)
    print("Example 2: 4K Video Processing")
    print("="*60)

    print("\nFor 4K (3840x2160) video:")
    print("  python style_transfer_main.py video \\")
    print("    input_4k.mp4 output_4k.mp4 \\")
    print("    --max-size 3840 \\")
    print("    --model-path model.pth")

    print("\nGPU Requirements for 4K:")
    print("  • Minimum: 10GB VRAM (RTX 3080)")
    print("  • Recommended: 16GB VRAM (RTX 4080)")
    print("  • RAM: 32GB system memory")
    print("  • Expected FPS: 10-15 on RTX 3080")


def example_video_formats():
    """Example: Different video formats"""
    print("\n" + "="*60)
    print("Example 3: Multiple Export Formats")
    print("="*60)

    formats = [
        ("MP4", "mp4v", "Standard MP4 format"),
        ("AVI", "XVID", "AVI with Xvid codec"),
        ("MOV", "avc1", "QuickTime MOV with H.264"),
        ("GIF", "gif", "Animated GIF"),
    ]

    print("\nSupported formats:")
    for format_name, codec, description in formats:
        print(f"  {format_name:6} - {description}")

    print("\nExport to different format:")
    print("  python style_transfer_main.py export \\")
    print("    input.mp4 output.mov \\")
    print("    --codec avc1 \\")
    print("    --quality 18")

    print("\nCreate GIF from video:")
    print("  python style_transfer_main.py export \\")
    print("    input.mp4 output.gif \\")
    print("    --format gif \\")
    print("    --fps 10 \\")
    print("    --max-size 480 \\")
    print("    --duration 10")


def example_temporal_smoothing():
    """Example: Temporal smoothing"""
    print("\n" + "="*60)
    print("Example 4: Temporal Smoothing")
    print("="*60)

    print("\nTemporal smoothing reduces flickering between frames")

    print("\nWith smoothing (default):")
    print("  python style_transfer_main.py video \\")
    print("    input.mp4 output.mp4 \\")
    print("    --model-path model.pth")

    print("\nWithout smoothing:")
    print("  python style_transfer_main.py video \\")
    print("    input.mp4 output.mp4 \\")
    print("    --model-path model.pth \\")
    print("    --no-smoothing")

    print("\nPython API:")
    print("  stats = processor.process_video(")
    print("      'input.mp4', 'output.mp4',")
    print("      use_temporal_smoothing=True  # Smooth transitions")
    print("  )")


def example_resolution_presets():
    """Example: Resolution presets"""
    print("\n" + "="*60)
    print("Example 5: Resolution Presets")
    print("="*60)

    resolutions = [
        ("SD", 640, "Fast processing, low quality"),
        ("HD", 1280, "Good balance"),
        ("Full HD", 1920, "High quality, moderate speed"),
        ("2K", 2560, "Very high quality"),
        ("4K", 3840, "Ultra high quality, slow"),
    ]

    print("\nRecommended max sizes:")
    for name, size, description in resolutions:
        print(f"  {name:10} (max-size={size:4}) - {description}")

    print("\nExample commands:")
    print("\n  # HD quality")
    print("  python style_transfer_main.py video input.mp4 output.mp4 --max-size 1280")

    print("\n  # 4K quality")
    print("  python style_transfer_main.py video input.mp4 output.mp4 --max-size 3840")


def example_batch_videos():
    """Example: Process multiple videos"""
    print("\n" + "="*60)
    print("Example 6: Batch Video Processing")
    print("="*60)

    print("\nBash script to process multiple videos:")
    print("""
#!/bin/bash
for video in ./input_videos/*.mp4; do
    filename=$(basename "$video" .mp4)
    python style_transfer_main.py video \\
        "$video" \\
        "./output_videos/${filename}_styled.mp4" \\
        --model-path model.pth \\
        --max-size 1920
done
    """)

    print("\nPython script:")
    print("""
from pathlib import Path
from neural_style_transfer import VideoStyleTransfer

processor = VideoStyleTransfer(model_path='model.pth')

input_dir = Path('./input_videos')
output_dir = Path('./output_videos')
output_dir.mkdir(exist_ok=True)

for video_file in input_dir.glob('*.mp4'):
    output_file = output_dir / f"{video_file.stem}_styled.mp4"
    print(f"Processing: {video_file.name}")

    stats = processor.process_video(
        str(video_file),
        str(output_file),
        max_size=1920
    )

    print(f"  Completed at {stats['avg_fps']:.2f} FPS")
    """)


def example_estimate_time():
    """Example: Estimate processing time"""
    print("\n" + "="*60)
    print("Example 7: Estimate Processing Time")
    print("="*60)

    print("\nPython API:")
    print("""
from neural_style_transfer import VideoStyleTransfer

processor = VideoStyleTransfer()

# Estimate processing time
estimate = processor.estimate_processing_time('video.mp4')

print(f"Video: {estimate['resolution']}")
print(f"Total frames: {estimate['total_frames']}")
print(f"Estimated FPS: {estimate['estimated_fps']}")
print(f"Estimated time: {estimate['estimated_time_formatted']}")
    """)


def performance_tips():
    """Performance optimization tips"""
    print("\n" + "="*60)
    print("Performance Optimization Tips")
    print("="*60)

    tips = {
        "GPU Optimization": [
            "Close other GPU-intensive applications",
            "Use CUDA 11.8 or newer",
            "Enable GPU performance mode in Windows",
            "Monitor GPU temperature"
        ],
        "Quality vs Speed": [
            "Lower resolution = faster processing",
            "ResNet model faster than VGG19",
            "Disable temporal smoothing for speed",
            "Use compressed codecs for output"
        ],
        "Memory Management": [
            "Process shorter clips for 4K",
            "Split long videos into chunks",
            "Clear GPU cache between runs",
            "Close browser and other apps"
        ],
        "Windows 11 Specific": [
            "Disable Windows animations",
            "Set power plan to 'High Performance'",
            "Update GPU drivers",
            "Use SSD for video files"
        ]
    }

    for category, tip_list in tips.items():
        print(f"\n{category}:")
        for tip in tip_list:
            print(f"  • {tip}")


def main():
    """Run all video processing examples"""
    print("\n" + "#"*60)
    print("# Neural Style Transfer - Video Processing Guide")
    print("#"*60)

    example_basic_video()
    example_4k_video()
    example_video_formats()
    example_temporal_smoothing()
    example_resolution_presets()
    example_batch_videos()
    example_estimate_time()
    performance_tips()

    print("\n" + "="*60)
    print("Video Processing Guide Complete!")
    print("="*60)
    print("\nQuick commands:")
    print("  • Process video: python style_transfer_main.py video input.mp4 output.mp4")
    print("  • 4K video: python style_transfer_main.py video input.mp4 output.mp4 --max-size 3840")
    print("  • Create GIF: python style_transfer_main.py export input.mp4 out.gif --format gif")
    print("\n")


if __name__ == '__main__':
    main()
