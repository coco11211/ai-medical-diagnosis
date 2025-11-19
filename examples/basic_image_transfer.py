"""
Basic Image Style Transfer Example

This example demonstrates how to apply style transfer to a single image
using both optimization-based and fast transfer methods.
"""

import os
from pathlib import Path

from nst_suite.core.transfer import OptimizationBasedTransfer, FastStyleTransfer
from nst_suite.utils.image import load_image, save_image
from nst_suite.utils.cuda import print_cuda_info


def main():
    """Run basic image style transfer example."""

    # Print system information
    print("="*60)
    print("Basic Image Style Transfer Example")
    print("="*60)
    print_cuda_info()

    # Setup paths
    content_path = "content.jpg"
    style_path = "style.jpg"
    output_dir = "./outputs"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if images exist
    if not os.path.exists(content_path):
        print(f"\nError: Content image not found at {content_path}")
        print("Please provide a content image.")
        return

    if not os.path.exists(style_path):
        print(f"\nError: Style image not found at {style_path}")
        print("Please provide a style image.")
        return

    # Load images
    print(f"\nLoading images...")
    content_image = load_image(content_path, max_size=512)
    style_image = load_image(style_path, max_size=512)

    print(f"Content image shape: {content_image.shape}")
    print(f"Style image shape: {style_image.shape}")

    # Method 1: Optimization-based transfer (high quality, slower)
    print("\n" + "="*60)
    print("Method 1: Optimization-based Transfer")
    print("="*60)

    transfer = OptimizationBasedTransfer(
        content_weight=1.0,
        style_weight=1e6,
        tv_weight=1e-3
    )

    print("Processing... (this may take a few minutes)")
    stylized_optimization = transfer.transfer(
        content_image,
        style_image,
        num_steps=300,
        learning_rate=0.03,
        init_method='content',
        show_progress=True
    )

    output_path = os.path.join(output_dir, "stylized_optimization.jpg")
    save_image(stylized_optimization, output_path)
    print(f"\nSaved result to: {output_path}")

    # Method 2: Fast transfer (requires pre-trained model)
    print("\n" + "="*60)
    print("Method 2: Fast Transfer")
    print("="*60)

    model_path = "./checkpoints/final_model.pth"

    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")

        fast_transfer = FastStyleTransfer(model_path=model_path)

        print("Processing...")
        stylized_fast = fast_transfer.transfer(content_image)

        output_path = os.path.join(output_dir, "stylized_fast.jpg")
        save_image(stylized_fast, output_path)
        print(f"Saved result to: {output_path}")

    else:
        print(f"Fast transfer model not found at {model_path}")
        print("To use fast transfer:")
        print("1. Train a model: nst train style.jpg ./dataset")
        print("2. Or download a pre-trained model")

    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60)
    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()
