"""
Batch Processing Example

Process multiple images and videos efficiently.
"""

import os
from nst_suite.batch.processor import BatchProcessor, MultiStyleBatchProcessor


def main():
    """Run batch processing examples."""

    print("="*60)
    print("Batch Processing Example")
    print("="*60)

    # Setup paths
    model_path = "./checkpoints/final_model.pth"
    input_dir = "./input_images"
    output_dir = "./outputs/batch"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train a model first:")
        print("  nst train style.jpg ./dataset")
        return

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"\nError: Input directory not found: {input_dir}")
        print("Creating example directory...")
        os.makedirs(input_dir, exist_ok=True)
        print(f"Please add images to: {input_dir}")
        return

    # Example 1: Batch process images
    print("\n" + "="*60)
    print("Example 1: Batch Image Processing")
    print("="*60)

    processor = BatchProcessor(
        model_path=model_path,
        num_workers=4
    )

    print(f"Processing all images in: {input_dir}")

    processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=True,
        batch_size=8,
        show_progress=True
    )

    print(f"Results saved to: {output_dir}")

    # Example 2: Process with multiple styles
    print("\n" + "="*60)
    print("Example 2: Multi-Style Processing")
    print("="*60)

    # Check if multiple models exist
    model_paths = [
        "./checkpoints/style1.pth",
        "./checkpoints/style2.pth",
        "./checkpoints/style3.pth"
    ]

    available_models = [p for p in model_paths if os.path.exists(p)]

    if len(available_models) > 1:
        print(f"Found {len(available_models)} style models")

        multi_processor = MultiStyleBatchProcessor(
            model_paths=available_models
        )

        # Process single image with all styles
        test_image = os.path.join(input_dir, "test.jpg")

        if os.path.exists(test_image):
            output_styles_dir = "./outputs/multi_style"
            os.makedirs(output_styles_dir, exist_ok=True)

            print(f"Applying all styles to: {test_image}")

            multi_processor.process_with_all_styles(
                input_path=test_image,
                output_dir=output_styles_dir,
                style_names=["style1", "style2", "style3"]
            )

            print(f"Results saved to: {output_styles_dir}")

            # Create style comparison grid
            grid_output = "./outputs/style_grid.jpg"

            print("\nCreating style comparison grid...")

            test_images = [
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if f.lower().endswith(('.jpg', '.png'))
            ][:3]  # Use first 3 images

            if len(test_images) > 0:
                multi_processor.create_style_grid(
                    input_paths=test_images,
                    output_path=grid_output
                )

                print(f"Grid saved to: {grid_output}")

        else:
            print(f"Test image not found: {test_image}")

    else:
        print("Multi-style example requires multiple trained models")
        print("Train additional models with different styles:")
        print("  nst train style1.jpg ./dataset --output-dir ./checkpoints/style1")
        print("  nst train style2.jpg ./dataset --output-dir ./checkpoints/style2")

    print("\n" + "="*60)
    print("Batch Processing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
