"""
Custom Training Pipeline for Neural Style Transfer
Train custom style transfer models on your own style images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, List
import logging
from tqdm import tqdm
import time
import json

from ..models.vgg19_model import VGG19StyleTransfer
from ..models.resnet_model import ResNetStyleTransfer, StyleTransferLoss
from ..utils.gpu_utils import GPUManager
from ..utils.image_utils import ImageProcessor


class ContentDataset(Dataset):
    """
    Dataset for loading content images during training
    """

    def __init__(self, image_dir: str, size: int = 256):
        """
        Initialize dataset

        Args:
            image_dir: Directory containing content images
            size: Image size for training
        """
        self.image_dir = Path(image_dir)
        self.size = size

        # Find all images
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_paths.extend(self.image_dir.glob(f"**/*{ext}"))

        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomCrop(size),
            transforms.ToTensor()
        ])

        logging.info(f"Loaded {len(self.image_paths)} content images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


class StyleTransferTrainer:
    """
    Trainer for custom style transfer models
    """

    def __init__(self,
                 style_image_path: str,
                 content_dataset_path: str,
                 output_dir: str = './trained_models',
                 device: Optional[str] = None):
        """
        Initialize trainer

        Args:
            style_image_path: Path to style image
            content_dataset_path: Path to content images directory
            output_dir: Directory to save trained models
            device: Device to train on
        """
        self.gpu_manager = GPUManager()
        self.device = device or self.gpu_manager.get_device()

        # Load style image
        self.style_image = Image.open(style_image_path).convert('RGB')
        logging.info(f"Loaded style image: {style_image_path}")

        # Create content dataset
        self.content_dataset = ContentDataset(content_dataset_path)

        # Initialize models
        self.style_model = ResNetStyleTransfer(device=self.device)
        self.vgg = VGG19StyleTransfer(device=self.device)

        # Extract style features
        self._extract_style_features()

        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')

        logging.info(f"Trainer initialized on {self.device}")

    def _extract_style_features(self):
        """Extract and cache style features"""
        style_tensor = self.vgg.preprocess_image(self.style_image)

        with torch.no_grad():
            _, style_features = self.vgg.extract_features(style_tensor)

            # Compute Gram matrices
            self.style_grams = {
                layer: self.vgg.gram_matrix(features)
                for layer, features in style_features.items()
            }

        logging.info("Style features extracted")

    def train(self,
             epochs: int = 2,
             batch_size: int = 4,
             learning_rate: float = 1e-3,
             content_weight: float = 1.0,
             style_weight: float = 1e5,
             tv_weight: float = 1e-6,
             save_every: int = 500,
             log_every: int = 100) -> Dict:
        """
        Train style transfer model

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            tv_weight: Weight for total variation loss
            save_every: Save checkpoint every N iterations
            log_every: Log stats every N iterations

        Returns:
            Training statistics
        """
        # Create data loader
        dataloader = DataLoader(
            self.content_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Optimizer
        optimizer = optim.Adam(self.style_model.parameters(), lr=learning_rate)

        # Loss function
        criterion = StyleTransferLoss(content_weight, style_weight, tv_weight)

        # Training loop
        total_iterations = 0
        training_losses = []

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_losses = []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch_idx, content_images in enumerate(pbar):
                content_images = content_images.to(self.device)

                # Forward pass through style model
                styled_images = self.style_model(content_images)

                # Extract features
                with torch.no_grad():
                    content_features, _ = self.vgg.extract_features(content_images)

                styled_content, styled_style = self.vgg.extract_features(styled_images)

                # Compute losses
                content_loss = criterion.content_loss(
                    styled_content['conv4_2'],
                    content_features['conv4_2']
                ) * content_weight

                style_loss = 0.0
                for layer in self.vgg.STYLE_LAYERS:
                    style_loss += criterion.style_loss(
                        styled_style[layer],
                        self.style_grams[layer]
                    )
                style_loss *= style_weight

                tv_loss = criterion.total_variation_loss(styled_images) * tv_weight

                total_loss = content_loss + style_loss + tv_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Track loss
                loss_value = total_loss.item()
                epoch_losses.append(loss_value)
                training_losses.append(loss_value)

                total_iterations += 1

                # Update progress bar
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f'{loss_value:.2f}',
                        'c_loss': f'{content_loss.item():.2f}',
                        's_loss': f'{style_loss.item():.2e}'
                    })

                # Log
                if total_iterations % log_every == 0:
                    logging.info(
                        f"Iteration {total_iterations}: "
                        f"Total={loss_value:.2f}, "
                        f"Content={content_loss.item():.2f}, "
                        f"Style={style_loss.item():.2e}, "
                        f"TV={tv_loss.item():.2e}"
                    )

                # Save checkpoint
                if total_iterations % save_every == 0:
                    self.save_checkpoint(
                        epoch,
                        total_iterations,
                        loss_value,
                        optimizer
                    )

            # Epoch summary
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logging.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.2f}")

            # Save best model
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.save_model('best_model.pth')
                logging.info(f"Saved best model with loss: {avg_epoch_loss:.2f}")

        # Save final model
        self.save_model('final_model.pth')

        # Training statistics
        stats = {
            'total_epochs': epochs,
            'total_iterations': total_iterations,
            'final_loss': training_losses[-1] if training_losses else 0,
            'best_loss': self.best_loss,
            'avg_loss': sum(training_losses) / len(training_losses) if training_losses else 0
        }

        # Save training stats
        self._save_training_stats(stats)

        logging.info("Training completed!")

        return stats

    def save_model(self, filename: str):
        """Save model weights"""
        save_path = self.output_dir / filename
        torch.save(self.style_model.state_dict(), save_path)
        logging.info(f"Model saved: {save_path}")

    def save_checkpoint(self, epoch: int, iteration: int, loss: float, optimizer):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': self.style_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        checkpoint_path = self.output_dir / f'checkpoint_iter_{iteration}.pth'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.style_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        logging.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def _save_training_stats(self, stats: Dict):
        """Save training statistics to JSON"""
        stats_path = self.output_dir / 'training_stats.json'

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logging.info(f"Training stats saved: {stats_path}")

    def test_model(self, test_image_path: str, output_path: str):
        """
        Test trained model on an image

        Args:
            test_image_path: Path to test image
            output_path: Path to save output
        """
        self.style_model.eval()

        test_image = Image.open(test_image_path).convert('RGB')

        with torch.no_grad():
            styled_image = self.style_model.stylize_image(test_image)

        styled_image.save(output_path)
        logging.info(f"Test output saved: {output_path}")


class QuickTrainer:
    """
    Quick training setup for fast experimentation
    """

    @staticmethod
    def train_quick(style_image_path: str,
                   content_dir: str,
                   output_model_path: str,
                   epochs: int = 1) -> str:
        """
        Quick training with default settings

        Args:
            style_image_path: Path to style image
            content_dir: Directory with content images
            output_model_path: Where to save trained model
            epochs: Number of epochs

        Returns:
            Path to trained model
        """
        trainer = StyleTransferTrainer(
            style_image_path,
            content_dir,
            output_dir=str(Path(output_model_path).parent)
        )

        trainer.train(epochs=epochs, batch_size=4)
        trainer.save_model(Path(output_model_path).name)

        return output_model_path
