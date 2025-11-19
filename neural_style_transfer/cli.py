"""
Neural Style Transfer CLI
Command-line interface for the neural style transfer suite
"""

import argparse
import sys
import logging
from pathlib import Path

from .core.style_transfer import StyleTransfer
from .core.video_processor import VideoStyleTransfer, VideoExporter
from .core.webcam_processor import WebcamStyleTransfer, DualViewWebcam
from .core.batch_processor import BatchProcessor, SmartBatchProcessor
from .core.trainer import StyleTransferTrainer, QuickTrainer
from .core.style_library import StyleLibrary, PresetStyles
from .utils.gpu_utils import GPUManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_image(args):
    """Process a single image"""
    print(f"Processing image: {args.input}")

    transfer = StyleTransfer(model_type=args.model)

    # Load images
    from PIL import Image
    content_image = Image.open(args.input)

    if args.model == 'vgg19':
        if not args.style:
            print("Error: --style required for VGG19 model")
            sys.exit(1)

        style_image = Image.open(args.style)

        output = transfer.transfer_style_optimization(
            content_image,
            style_image,
            num_steps=args.steps,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            size=args.size
        )
    else:
        output = transfer.transfer_style_fast(content_image)

    output.save(args.output)
    print(f"Output saved: {args.output}")


def cmd_video(args):
    """Process a video"""
    print(f"Processing video: {args.input}")

    processor = VideoStyleTransfer(model_path=args.model_path)

    stats = processor.process_video(
        args.input,
        args.output,
        max_size=args.max_size,
        use_temporal_smoothing=not args.no_smoothing,
        fps=args.fps,
        codec=args.codec
    )

    print("\nProcessing complete!")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Average FPS: {stats['avg_fps']:.2f}")
    print(f"  Total time: {stats['total_time_s']:.2f}s")
    print(f"  Output: {args.output}")


def cmd_webcam(args):
    """Start webcam mode"""
    print("Starting webcam mode...")

    if args.dual_view:
        webcam = DualViewWebcam(camera_id=args.camera)
        webcam.start(resolution=(args.width, args.height))
    else:
        webcam = WebcamStyleTransfer(camera_id=args.camera)
        webcam.start(
            resolution=(args.width, args.height),
            save_output=args.save,
            output_path=args.output
        )


def cmd_batch(args):
    """Batch process images"""
    print(f"Batch processing: {args.input_dir}")

    if args.auto:
        processor = SmartBatchProcessor()
        stats = processor.auto_process(args.input_dir, args.output_dir)
    else:
        processor = BatchProcessor(model_path=args.model_path)

        if args.parallel and processor.gpu_manager.gpu_count > 1:
            stats = processor.process_with_multiple_gpus(
                list(Path(args.input_dir).glob('*.*')),
                args.output_dir
            )
        else:
            stats = processor.process_directory(
                args.input_dir,
                args.output_dir
            )

    print("\nBatch processing complete!")
    print(f"  Total: {stats['total']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    if 'avg_time' in stats:
        print(f"  Avg time per image: {stats['avg_time']:.2f}s")


def cmd_train(args):
    """Train a custom model"""
    print(f"Training on style: {args.style_image}")
    print(f"Content dataset: {args.content_dir}")

    if args.quick:
        model_path = QuickTrainer.train_quick(
            args.style_image,
            args.content_dir,
            args.output_model,
            epochs=args.epochs
        )
        print(f"Model trained: {model_path}")
    else:
        trainer = StyleTransferTrainer(
            args.style_image,
            args.content_dir,
            output_dir=str(Path(args.output_model).parent)
        )

        stats = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        print("\nTraining complete!")
        print(f"  Final loss: {stats['final_loss']:.2f}")
        print(f"  Best loss: {stats['best_loss']:.2f}")
        print(f"  Model saved: {args.output_model}")


def cmd_library(args):
    """Manage style library"""
    library = StyleLibrary(args.library_path)

    if args.action == 'list':
        styles = library.list_styles(tags=args.tags)
        print(f"\nFound {len(styles)} styles:")
        for style in styles:
            print(f"  {style['id']}: {style['name']}")
            print(f"    Tags: {', '.join(style['tags'])}")
            print(f"    Model: {'Yes' if style['model'] else 'No'}")
            print()

    elif args.action == 'add':
        library.add_style(
            args.style_id,
            args.style_image,
            model_path=args.model_path,
            name=args.name,
            description=args.description,
            tags=args.tags
        )
        print(f"Style '{args.style_id}' added to library")

    elif args.action == 'remove':
        library.remove_style(args.style_id)
        print(f"Style '{args.style_id}' removed from library")

    elif args.action == 'export':
        library.export_library(args.export_path)
        print(f"Library exported to {args.export_path}")

    elif args.action == 'import':
        library.import_library(args.import_path)
        print(f"Library imported from {args.import_path}")

    elif args.action == 'presets':
        presets = PresetStyles.list_presets()
        print(f"\nAvailable presets ({len(presets)}):")
        for preset_id in presets:
            info = PresetStyles.get_preset_info(preset_id)
            print(f"  {preset_id}: {info['name']}")
            print(f"    {info['description']}")
            print()


def cmd_gpu_info(args):
    """Display GPU information"""
    gpu_manager = GPUManager()
    print("\n" + "="*60)
    print("GPU Information")
    print("="*60)
    print(gpu_manager.get_cuda_info())


def cmd_export(args):
    """Export video to different format"""
    print(f"Exporting video: {args.input}")

    if args.format == 'gif':
        VideoExporter.create_gif(
            args.input,
            args.output,
            max_size=args.max_size,
            fps=args.fps,
            duration=args.duration
        )
    else:
        VideoExporter.export_video(
            args.input,
            args.output,
            codec=args.codec,
            crf=args.quality
        )

    print(f"Exported: {args.output}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Neural Style Transfer Suite - Real-time video processing with GPU acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Image command
    parser_image = subparsers.add_parser('image', help='Process a single image')
    parser_image.add_argument('input', help='Input image path')
    parser_image.add_argument('output', help='Output image path')
    parser_image.add_argument('--style', help='Style image path (for VGG19)')
    parser_image.add_argument('--model', choices=['vgg19', 'resnet'], default='vgg19', help='Model type')
    parser_image.add_argument('--steps', type=int, default=300, help='Optimization steps (VGG19)')
    parser_image.add_argument('--content-weight', type=float, default=1.0, help='Content weight')
    parser_image.add_argument('--style-weight', type=float, default=1000000.0, help='Style weight')
    parser_image.add_argument('--size', type=int, help='Max dimension size')
    parser_image.set_defaults(func=cmd_image)

    # Video command
    parser_video = subparsers.add_parser('video', help='Process a video file')
    parser_video.add_argument('input', help='Input video path')
    parser_video.add_argument('output', help='Output video path')
    parser_video.add_argument('--model-path', help='Path to pre-trained model')
    parser_video.add_argument('--max-size', type=int, help='Maximum dimension')
    parser_video.add_argument('--no-smoothing', action='store_true', help='Disable temporal smoothing')
    parser_video.add_argument('--fps', type=int, help='Output FPS')
    parser_video.add_argument('--codec', default='mp4v', help='Video codec')
    parser_video.set_defaults(func=cmd_video)

    # Webcam command
    parser_webcam = subparsers.add_parser('webcam', help='Real-time webcam processing')
    parser_webcam.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser_webcam.add_argument('--width', type=int, default=1280, help='Camera width')
    parser_webcam.add_argument('--height', type=int, default=720, help='Camera height')
    parser_webcam.add_argument('--save', action='store_true', help='Save output video')
    parser_webcam.add_argument('--output', help='Output video path')
    parser_webcam.add_argument('--dual-view', action='store_true', help='Show original and styled side-by-side')
    parser_webcam.set_defaults(func=cmd_webcam)

    # Batch command
    parser_batch = subparsers.add_parser('batch', help='Batch process multiple images')
    parser_batch.add_argument('input_dir', help='Input directory')
    parser_batch.add_argument('output_dir', help='Output directory')
    parser_batch.add_argument('--model-path', help='Path to pre-trained model')
    parser_batch.add_argument('--parallel', action='store_true', help='Use multi-GPU processing')
    parser_batch.add_argument('--auto', action='store_true', help='Auto-optimize processing')
    parser_batch.set_defaults(func=cmd_batch)

    # Train command
    parser_train = subparsers.add_parser('train', help='Train a custom style model')
    parser_train.add_argument('style_image', help='Style image path')
    parser_train.add_argument('content_dir', help='Content images directory')
    parser_train.add_argument('output_model', help='Output model path')
    parser_train.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser_train.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser_train.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser_train.add_argument('--quick', action='store_true', help='Quick training mode')
    parser_train.set_defaults(func=cmd_train)

    # Library command
    parser_library = subparsers.add_parser('library', help='Manage style library')
    parser_library.add_argument('action', choices=['list', 'add', 'remove', 'export', 'import', 'presets'])
    parser_library.add_argument('--library-path', default='./style_library', help='Library path')
    parser_library.add_argument('--style-id', help='Style ID')
    parser_library.add_argument('--style-image', help='Style image path')
    parser_library.add_argument('--model-path', help='Model path')
    parser_library.add_argument('--name', help='Style name')
    parser_library.add_argument('--description', help='Style description')
    parser_library.add_argument('--tags', nargs='+', help='Style tags')
    parser_library.add_argument('--export-path', help='Export path (zip file)')
    parser_library.add_argument('--import-path', help='Import path (zip file)')
    parser_library.set_defaults(func=cmd_library)

    # GPU info command
    parser_gpu = subparsers.add_parser('gpu-info', help='Display GPU information')
    parser_gpu.set_defaults(func=cmd_gpu_info)

    # Export command
    parser_export = subparsers.add_parser('export', help='Export video to different format')
    parser_export.add_argument('input', help='Input video path')
    parser_export.add_argument('output', help='Output path')
    parser_export.add_argument('--format', choices=['video', 'gif'], default='video')
    parser_export.add_argument('--codec', help='Video codec')
    parser_export.add_argument('--quality', type=int, default=23, help='Quality (CRF for video, 0-51)')
    parser_export.add_argument('--fps', type=int, default=10, help='FPS (for GIF)')
    parser_export.add_argument('--max-size', type=int, default=480, help='Max dimension (for GIF)')
    parser_export.add_argument('--duration', type=float, help='Max duration in seconds (for GIF)')
    parser_export.set_defaults(func=cmd_export)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
