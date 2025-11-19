"""Command-line interface for Neural Style Transfer Suite."""

import click
import os
import sys
from pathlib import Path

from .core.transfer import OptimizationBasedTransfer, FastStyleTransfer
from .video.processor import VideoStyleTransfer, RealtimeVideoProcessor
from .video.webcam import WebcamStyleTransfer
from .batch.processor import BatchProcessor, MultiStyleBatchProcessor
from .library.manager import StyleLibrary
from .training.train import StyleTrainer
from .utils.cuda import print_cuda_info
from .utils.image import load_image, save_image


@click.group()
@click.version_option(version='1.0.0')
def main():
    """
    Neural Style Transfer Suite - Comprehensive style transfer toolkit.

    A powerful suite for applying artistic styles to images and videos
    with support for real-time processing, GPU acceleration, and more.
    """
    pass


@main.command()
@click.argument('content_image', type=click.Path(exists=True))
@click.argument('style_image', type=click.Path(exists=True))
@click.argument('output_image', type=click.Path())
@click.option('--method', type=click.Choice(['optimization', 'fast']), default='optimization',
              help='Transfer method to use')
@click.option('--model', type=click.Path(exists=True), help='Pre-trained model for fast transfer')
@click.option('--steps', default=300, help='Number of optimization steps (optimization method only)')
@click.option('--content-weight', default=1.0, help='Content loss weight')
@click.option('--style-weight', default=1e6, help='Style loss weight')
@click.option('--max-size', default=512, help='Maximum image size')
def transfer(content_image, style_image, output_image, method, model, steps,
             content_weight, style_weight, max_size):
    """
    Apply style transfer to a single image.

    CONTENT_IMAGE: Path to content image
    STYLE_IMAGE: Path to style image
    OUTPUT_IMAGE: Path to save stylized image
    """
    click.echo(f"Loading images...")

    content = load_image(content_image, max_size=max_size)

    if method == 'optimization':
        click.echo(f"Using optimization-based transfer...")

        style = load_image(style_image, max_size=max_size)

        transfer_engine = OptimizationBasedTransfer(
            content_weight=content_weight,
            style_weight=style_weight
        )

        stylized = transfer_engine.transfer(
            content, style,
            num_steps=steps,
            show_progress=True
        )

    else:  # fast
        if not model:
            click.echo("Error: --model required for fast transfer method", err=True)
            sys.exit(1)

        click.echo(f"Using fast transfer with model: {model}")

        transfer_engine = FastStyleTransfer(model_path=model)
        stylized = transfer_engine.transfer(content)

    save_image(stylized, output_image)
    click.echo(f"Stylized image saved to: {output_image}")


@main.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.option('--model', type=click.Path(exists=True), required=True,
              help='Pre-trained style transfer model')
@click.option('--batch-size', default=4, help='Batch size for processing')
@click.option('--realtime', is_flag=True, help='Use real-time processing mode')
@click.option('--resize-factor', default=1.0, help='Resize factor for faster processing')
def video(input_video, output_video, model, batch_size, realtime, resize_factor):
    """
    Apply style transfer to video.

    INPUT_VIDEO: Path to input video
    OUTPUT_VIDEO: Path to save output video
    """
    if realtime:
        click.echo(f"Processing video in real-time mode...")
        processor = RealtimeVideoProcessor(
            model_path=model,
            resize_factor=resize_factor
        )
        processor.process_video_file(input_video, output_video)
    else:
        click.echo(f"Processing video with batch size {batch_size}...")
        processor = VideoStyleTransfer(model_path=model)
        processor.process_video(
            input_video,
            output_video,
            batch_size=batch_size
        )

    click.echo(f"Output saved to: {output_video}")


@main.command()
@click.option('--model', type=click.Path(exists=True), required=True,
              help='Pre-trained style transfer model')
@click.option('--camera', default=0, help='Camera device ID')
@click.option('--resize-factor', default=0.75, help='Resize factor for performance')
@click.option('--fps', default=30, help='Target frames per second')
@click.option('--save', type=click.Path(), help='Save output to video file')
def webcam(model, camera, resize_factor, fps, save):
    """
    Run real-time webcam style transfer.
    """
    click.echo(f"Starting webcam style transfer...")
    click.echo(f"Camera: {camera}, Target FPS: {fps}")
    click.echo(f"Press 'q' to quit")

    webcam_processor = WebcamStyleTransfer(
        model_path=model,
        camera_id=camera,
        resize_factor=resize_factor
    )

    webcam_processor.run(
        target_fps=fps,
        save_output=save
    )


@main.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--model', type=click.Path(exists=True), required=True,
              help='Pre-trained style transfer model')
@click.option('--batch-size', default=8, help='Batch size for processing')
@click.option('--recursive', is_flag=True, help='Process directories recursively')
def batch(input_dir, output_dir, model, batch_size, recursive):
    """
    Batch process multiple images.

    INPUT_DIR: Directory containing images
    OUTPUT_DIR: Directory to save output
    """
    click.echo(f"Batch processing images from: {input_dir}")

    processor = BatchProcessor(model_path=model)
    processor.process_directory(
        input_dir,
        output_dir,
        recursive=recursive,
        batch_size=batch_size
    )


@main.command()
@click.argument('style_image', type=click.Path(exists=True))
@click.argument('content_dir', type=click.Path(exists=True))
@click.option('--output-dir', default='./checkpoints', help='Directory to save checkpoints')
@click.option('--epochs', default=2, help='Number of training epochs')
@click.option('--batch-size', default=4, help='Batch size')
@click.option('--lr', default=1e-3, help='Learning rate')
@click.option('--image-size', default=256, help='Training image size')
@click.option('--style-weight', default=1e5, help='Style loss weight')
def train(style_image, content_dir, output_dir, epochs, batch_size, lr, image_size, style_weight):
    """
    Train a fast style transfer model.

    STYLE_IMAGE: Path to style image
    CONTENT_DIR: Directory containing content images for training
    """
    click.echo(f"Training fast style transfer model...")
    click.echo(f"Style image: {style_image}")
    click.echo(f"Content directory: {content_dir}")

    trainer = StyleTrainer(
        style_image_path=style_image,
        style_weight=style_weight
    )

    trainer.train(
        content_dir=content_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        image_size=image_size,
        save_dir=output_dir
    )

    click.echo(f"Training complete! Model saved to: {output_dir}")


@main.group()
def library():
    """Manage style library."""
    pass


@library.command('add')
@click.argument('name')
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--style-image', type=click.Path(exists=True), help='Style image')
@click.option('--description', help='Description')
@click.option('--author', help='Author name')
@click.option('--tags', help='Comma-separated tags')
@click.option('--library-dir', default='./style_library', help='Library directory')
def library_add(name, model_path, style_image, description, author, tags, library_dir):
    """Add a model to the style library."""
    lib = StyleLibrary(library_dir)

    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(',')]

    model_id = lib.add_model(
        name=name,
        model_path=model_path,
        style_image_path=style_image,
        description=description,
        author=author,
        tags=tag_list
    )

    click.echo(f"Added style '{name}' with ID: {model_id}")


@library.command('list')
@click.option('--tags', help='Filter by tags (comma-separated)')
@click.option('--search', help='Search query')
@click.option('--library-dir', default='./style_library', help='Library directory')
def library_list(tags, search, library_dir):
    """List all styles in the library."""
    lib = StyleLibrary(library_dir)

    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(',')]

    models = lib.list_models(tags=tag_list, search=search)

    if not models:
        click.echo("No styles found.")
        return

    click.echo(f"\nFound {len(models)} style(s):\n")

    for model in models:
        click.echo(f"  ID: {model.id}")
        click.echo(f"  Name: {model.name}")
        if model.description:
            click.echo(f"  Description: {model.description}")
        if model.tags:
            click.echo(f"  Tags: {', '.join(model.tags)}")
        click.echo()


@library.command('remove')
@click.argument('model_id')
@click.option('--library-dir', default='./style_library', help='Library directory')
@click.confirmation_option(prompt='Are you sure you want to remove this style?')
def library_remove(model_id, library_dir):
    """Remove a style from the library."""
    lib = StyleLibrary(library_dir)
    lib.remove_model(model_id)
    click.echo(f"Removed style: {model_id}")


@library.command('stats')
@click.option('--library-dir', default='./style_library', help='Library directory')
def library_stats(library_dir):
    """Show library statistics."""
    lib = StyleLibrary(library_dir)
    stats = lib.get_stats()

    click.echo("\nLibrary Statistics:")
    click.echo(f"  Total models: {stats['total_models']}")
    click.echo(f"  Total tags: {stats['total_tags']}")
    click.echo(f"  Library size: {stats['library_size_mb']:.2f} MB")

    if stats['tags']:
        click.echo(f"  Tags: {', '.join(stats['tags'])}")


@main.command()
def info():
    """Show system and CUDA information."""
    click.echo("\n=== Neural Style Transfer Suite ===")
    click.echo("Version: 1.0.0")
    click.echo("\nSystem Information:")

    print_cuda_info()


if __name__ == '__main__':
    main()
