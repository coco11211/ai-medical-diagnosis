"""
Quick Start Example for Neural Style Transfer
Demonstrates basic usage of the style transfer suite
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural_style_transfer import StyleTransfer, GPUManager
from PIL import Image


def example_1_basic_image_transfer():
    """Example 1: Basic image style transfer"""
    print("\n" + "="*60)
    print("Example 1: Basic Image Style Transfer (VGG19)")
    print("="*60)

    # Initialize
    transfer = StyleTransfer(model_type='vgg19')

    # Check GPU
    gpu_manager = GPUManager()
    print(f"Using device: {gpu_manager.get_device()}")

    # Note: You need to provide your own images
    # Create dummy images for demonstration
    content = Image.new('RGB', (512, 512), color='white')
    style = Image.new('RGB', (512, 512), color='blue')

    print("\nProcessing image...")
    print("Note: Replace with your actual images:")
    print("  content = Image.open('your_content.jpg')")
    print("  style = Image.open('your_style.jpg')")

    # Process
    # output = transfer.transfer(content, style, num_steps=100)
    # output.save('output.jpg')

    print("\nProcessing complete (example only)")


def example_2_fast_transfer():
    """Example 2: Fast style transfer using ResNet"""
    print("\n" + "="*60)
    print("Example 2: Fast Style Transfer (ResNet)")
    print("="*60)

    # Initialize with ResNet for speed
    transfer = StyleTransfer(model_type='resnet')

    print("\nNote: ResNet requires a pre-trained model")
    print("Train a model first using:")
    print("  python style_transfer_main.py train style.jpg ./content_dir model.pth")

    print("\nThen process images:")
    # content = Image.open('content.jpg')
    # output = transfer.transfer_style_fast(content)
    # output.save('output_fast.jpg')


def example_3_gpu_info():
    """Example 3: Check GPU information"""
    print("\n" + "="*60)
    print("Example 3: GPU Information")
    print("="*60)

    gpu_manager = GPUManager()
    info = gpu_manager.get_gpu_info()

    print(f"\nGPU Available: {info['available']}")
    print(f"Number of GPUs: {info['count']}")

    if info['available']:
        for device in info['devices']:
            print(f"\nGPU {device['id']}: {device['name']}")
            print(f"  Total Memory: {device['total_memory_gb']:.2f} GB")
            print(f"  Compute Capability: {device['compute_capability']}")

        # Memory stats
        stats = gpu_manager.get_memory_stats()
        print(f"\nMemory Usage:")
        print(f"  Allocated: {stats['allocated_gb']:.2f} GB")
        print(f"  Free: {stats['free_gb']:.2f} GB")
        print(f"  Total: {stats['total_gb']:.2f} GB")


def example_4_batch_processing():
    """Example 4: Batch process multiple images"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)

    from neural_style_transfer import BatchProcessor

    print("\nBatch processing example:")
    print("  processor = BatchProcessor()")
    print("  stats = processor.process_directory('./input', './output')")
    print("\nThis will process all images in the input directory")


def example_5_video_processing():
    """Example 5: Process a video"""
    print("\n" + "="*60)
    print("Example 5: Video Processing")
    print("="*60)

    from neural_style_transfer import VideoStyleTransfer

    print("\nVideo processing example:")
    print("  processor = VideoStyleTransfer(model_path='model.pth')")
    print("  stats = processor.process_video('input.mp4', 'output.mp4')")
    print("\nSupports resolutions up to 4K with GPU acceleration")


def example_6_webcam():
    """Example 6: Real-time webcam processing"""
    print("\n" + "="*60)
    print("Example 6: Webcam Processing")
    print("="*60)

    print("\nWebcam processing example:")
    print("  from neural_style_transfer import WebcamStyleTransfer")
    print("  webcam = WebcamStyleTransfer()")
    print("  webcam.start()")
    print("\nControls:")
    print("  Q/ESC - Quit")
    print("  S - Toggle smoothing")
    print("  F - Toggle FPS")
    print("  SPACE - Screenshot")


def example_7_style_interpolation():
    """Example 7: Style interpolation"""
    print("\n" + "="*60)
    print("Example 7: Style Interpolation")
    print("="*60)

    from neural_style_transfer.utils import StyleInterpolator

    print("\nStyle interpolation example:")
    print("  interpolator = StyleInterpolator()")
    print("  blended = interpolator.blend_styles(")
    print("      [style1_features, style2_features],")
    print("      weights=[0.7, 0.3]")
    print("  )")
    print("\nCreate smooth transitions between multiple styles")


def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("# Neural Style Transfer Suite - Quick Start Examples")
    print("#"*60)

    # Run examples
    example_1_basic_image_transfer()
    example_2_fast_transfer()
    example_3_gpu_info()
    example_4_batch_processing()
    example_5_video_processing()
    example_6_webcam()
    example_7_style_interpolation()

    print("\n" + "="*60)
    print("Quick Start Guide Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check GPU: python style_transfer_main.py gpu-info")
    print("2. Process image: python style_transfer_main.py image input.jpg output.jpg --style style.jpg")
    print("3. Try webcam: python style_transfer_main.py webcam")
    print("4. Read full docs: neural_style_transfer/README.md")
    print("\n")


if __name__ == '__main__':
    main()
