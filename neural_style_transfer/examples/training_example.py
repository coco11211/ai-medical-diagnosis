"""
Training Example for Neural Style Transfer
Shows how to train custom style transfer models
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural_style_transfer.core.trainer import StyleTransferTrainer


def train_custom_model():
    """
    Example: Train a custom style transfer model

    Requirements:
    - A style image (e.g., starry_night.jpg)
    - A dataset of content images (e.g., MS COCO dataset)
    """

    print("="*60)
    print("Training Custom Style Transfer Model")
    print("="*60)

    # Paths (modify these to your actual paths)
    STYLE_IMAGE = "path/to/your/style_image.jpg"
    CONTENT_DATASET = "path/to/content_images/"
    OUTPUT_DIR = "./trained_models"

    print(f"\nConfiguration:")
    print(f"  Style image: {STYLE_IMAGE}")
    print(f"  Content dataset: {CONTENT_DATASET}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Note: This is a demonstration. Replace with actual paths
    print("\n" + "-"*60)
    print("DEMO MODE - Replace with your actual image paths")
    print("-"*60)
    print("\nTo actually train a model:")
    print("1. Download a content image dataset (e.g., MS COCO)")
    print("2. Choose a style image")
    print("3. Update the paths above")
    print("4. Run this script")

    print("\nExample using CLI:")
    print("  python style_transfer_main.py train \\")
    print("    starry_night.jpg \\")
    print("    ./coco_dataset \\")
    print("    ./models/starry_night.pth \\")
    print("    --epochs 2 \\")
    print("    --batch-size 4")

    # Uncomment to actually train (requires valid paths)
    """
    trainer = StyleTransferTrainer(
        style_image_path=STYLE_IMAGE,
        content_dataset_path=CONTENT_DATASET,
        output_dir=OUTPUT_DIR
    )

    # Train the model
    stats = trainer.train(
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        content_weight=1.0,
        style_weight=1e5,
        save_every=500,
        log_every=100
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final loss: {stats['final_loss']:.2f}")
    print(f"Best loss: {stats['best_loss']:.2f}")
    print(f"Total iterations: {stats['total_iterations']}")

    # Test the trained model
    print("\nTesting trained model...")
    trainer.test_model(
        test_image_path="test_input.jpg",
        output_path="test_output.jpg"
    )
    print("Test output saved: test_output.jpg")
    """


def download_dataset_info():
    """
    Information about downloading training datasets
    """
    print("\n" + "="*60)
    print("Training Dataset Information")
    print("="*60)

    print("\nRecommended Content Image Datasets:")
    print("\n1. MS COCO Dataset")
    print("   - URL: https://cocodataset.org/")
    print("   - Size: ~25GB (train2017)")
    print("   - Images: 118,000+")
    print("   - Best for: General purpose training")

    print("\n2. ImageNet")
    print("   - URL: https://www.image-net.org/")
    print("   - Size: ~150GB")
    print("   - Images: 1.2M+")
    print("   - Best for: High-quality diverse content")

    print("\n3. Places365")
    print("   - URL: http://places2.csail.mit.edu/")
    print("   - Size: ~105GB")
    print("   - Images: 1.8M+")
    print("   - Best for: Landscape and scene content")

    print("\n4. DIV2K (smaller dataset)")
    print("   - URL: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
    print("   - Size: ~5GB")
    print("   - Images: 1000")
    print("   - Best for: Quick testing")

    print("\nQuick Start with COCO:")
    print("  1. Download: wget http://images.cocodataset.org/zips/train2017.zip")
    print("  2. Extract: unzip train2017.zip")
    print("  3. Train: python style_transfer_main.py train style.jpg ./train2017 model.pth")


def training_tips():
    """
    Tips for training better models
    """
    print("\n" + "="*60)
    print("Training Tips")
    print("="*60)

    tips = [
        {
            "title": "1. Choose Good Style Images",
            "tips": [
                "High resolution (at least 512x512)",
                "Clear style patterns",
                "Good color composition",
                "Avoid too much detail that might not transfer well"
            ]
        },
        {
            "title": "2. Content Dataset",
            "tips": [
                "Use diverse content images",
                "Mix of objects, landscapes, people",
                "At least 1000 images recommended",
                "Images should be representative of your use case"
            ]
        },
        {
            "title": "3. Training Parameters",
            "tips": [
                "Start with 2 epochs for testing",
                "Batch size: 4 for 8GB GPU, 8 for 16GB GPU",
                "Learning rate: 1e-3 is a good default",
                "Style weight: 1e5 to 1e6",
                "Content weight: 1.0"
            ]
        },
        {
            "title": "4. GPU Memory",
            "tips": [
                "Reduce batch size if out of memory",
                "Lower image size during training",
                "Close other GPU applications",
                "Monitor GPU usage: nvidia-smi"
            ]
        },
        {
            "title": "5. Quality Checks",
            "tips": [
                "Test model every 500 iterations",
                "Compare with different content images",
                "Check for artifacts or over-stylization",
                "Adjust weights if needed"
            ]
        }
    ]

    for tip_group in tips:
        print(f"\n{tip_group['title']}")
        for tip in tip_group['tips']:
            print(f"  • {tip}")


def main():
    """Run training examples and info"""
    print("\n" + "#"*60)
    print("# Neural Style Transfer - Training Guide")
    print("#"*60)

    train_custom_model()
    download_dataset_info()
    training_tips()

    print("\n" + "="*60)
    print("Training Guide Complete!")
    print("="*60)
    print("\nReady to train? Follow these steps:")
    print("1. Download a content dataset (COCO recommended)")
    print("2. Choose or create a style image")
    print("3. Run: python style_transfer_main.py train style.jpg ./dataset model.pth")
    print("4. Wait for training to complete (may take hours)")
    print("5. Test your model on new images!")
    print("\n")


if __name__ == '__main__':
    main()
