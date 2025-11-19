"""Training module for fast style transfer models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm
import os
from pathlib import Path

from ..models.vgg import VGG19StyleTransfer, FastVGG19StyleTransfer
from ..utils.image import load_image, preprocess_image
from .dataset import StyleDataset


class StyleTrainer:
    """
    Trainer for fast style transfer models.
    """

    def __init__(
        self,
        style_image_path: str,
        content_weight: float = 1.0,
        style_weight: float = 1e5,
        tv_weight: float = 1e-6,
        device: Optional[torch.device] = None
    ):
        """
        Initialize style trainer.

        Args:
            style_image_path: Path to style image
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            tv_weight: Weight for total variation loss
            device: PyTorch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

        # Load style image
        style_image = load_image(style_image_path, max_size=512)
        self.style_tensor = preprocess_image(style_image, self.device)

        # Initialize models
        self.vgg = VGG19StyleTransfer(device=self.device)
        self.transfer_net = FastVGG19StyleTransfer(device=self.device)

        # Extract style features
        with torch.no_grad():
            self.style_features = self.vgg.extract_features(self.style_tensor)

        print(f"Style trainer initialized on {self.device}")

    def train(
        self,
        content_dir: str,
        num_epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        image_size: int = 256,
        save_dir: str = './checkpoints',
        save_interval: int = 500
    ):
        """
        Train the fast style transfer model.

        Args:
            content_dir: Directory containing content images
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            image_size: Size to resize images to
            save_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N iterations
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Create dataset and dataloader
        dataset = StyleDataset(content_dir, image_size=image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Optimizer
        optimizer = optim.Adam(self.transfer_net.parameters(), lr=learning_rate)

        # Training loop
        self.transfer_net.train()
        iteration = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            epoch_content_loss = 0
            epoch_style_loss = 0
            epoch_tv_loss = 0

            progress_bar = tqdm(dataloader)

            for batch_idx, content_images in enumerate(progress_bar):
                iteration += 1

                content_images = content_images.to(self.device)

                # Forward pass
                stylized_images = self.transfer_net(content_images)

                # Extract features
                content_features = self.vgg.extract_features(content_images)
                stylized_features = self.vgg.extract_features(stylized_images)

                # Compute losses
                content_loss = self.vgg.compute_content_loss(
                    content_features, stylized_features
                )

                # For style loss, compare with the single style image
                # We need to expand style features to match batch size
                batch_style_features = {}
                for layer, features in self.style_features.items():
                    batch_style_features[layer] = features.expand(
                        content_images.size(0), -1, -1, -1
                    )

                style_loss = self.vgg.compute_style_loss(
                    batch_style_features, stylized_features
                )

                tv_loss = self.vgg.compute_total_variation_loss(stylized_images)

                # Total loss
                total_loss = (
                    self.content_weight * content_loss +
                    self.style_weight * style_loss +
                    self.tv_weight * tv_loss
                )

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Update statistics
                epoch_content_loss += content_loss.item()
                epoch_style_loss += style_loss.item()
                epoch_tv_loss += tv_loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'content': f'{content_loss.item():.2f}',
                    'style': f'{style_loss.item():.2f}',
                    'tv': f'{tv_loss.item():.6f}'
                })

                # Save checkpoint
                if iteration % save_interval == 0:
                    checkpoint_path = os.path.join(
                        save_dir,
                        f'checkpoint_iter_{iteration}.pth'
                    )
                    self.save_checkpoint(checkpoint_path, epoch, iteration)

            # Print epoch statistics
            num_batches = len(dataloader)
            print(f"Epoch {epoch + 1} - "
                  f"Content: {epoch_content_loss/num_batches:.2f}, "
                  f"Style: {epoch_style_loss/num_batches:.2f}, "
                  f"TV: {epoch_tv_loss/num_batches:.6f}")

        # Save final model
        final_path = os.path.join(save_dir, 'final_model.pth')
        self.save_checkpoint(final_path, num_epochs, iteration)
        print(f"\nTraining complete! Model saved to {final_path}")

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        iteration: int
    ):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            iteration: Current iteration
        """
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': self.transfer_net.state_dict(),
            'content_weight': self.content_weight,
            'style_weight': self.style_weight,
            'tv_weight': self.tv_weight,
        }, path)

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.transfer_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")


def main():
    """Main training function for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description='Train fast style transfer model')
    parser.add_argument('--style-image', type=str, required=True,
                        help='Path to style image')
    parser.add_argument('--content-dir', type=str, required=True,
                        help='Directory containing content images')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--content-weight', type=float, default=1.0,
                        help='Content loss weight')
    parser.add_argument('--style-weight', type=float, default=1e5,
                        help='Style loss weight')
    parser.add_argument('--tv-weight', type=float, default=1e-6,
                        help='Total variation loss weight')

    args = parser.parse_args()

    # Initialize trainer
    trainer = StyleTrainer(
        style_image_path=args.style_image,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight
    )

    # Train
    trainer.train(
        content_dir=args.content_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
