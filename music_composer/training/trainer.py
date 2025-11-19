"""
Training pipeline for music generation models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
from tqdm import tqdm


class MusicDataset(Dataset):
    """Dataset for music training"""

    def __init__(
        self,
        sequences: List[np.ndarray],
        seq_length: int = 50,
        genres: Optional[List[int]] = None,
        moods: Optional[List[int]] = None
    ):
        self.sequences = sequences
        self.seq_length = seq_length
        self.genres = genres
        self.moods = moods

        # Create training samples
        self.samples = []
        for idx, seq in enumerate(sequences):
            genre = genres[idx] if genres else 0
            mood = moods[idx] if moods else 0

            for i in range(len(seq) - seq_length):
                input_seq = seq[i:i + seq_length]
                target_seq = seq[i + 1:i + seq_length + 1]
                self.samples.append({
                    'input': input_seq,
                    'target': target_seq,
                    'genre': genre,
                    'mood': mood
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input': torch.tensor(sample['input'], dtype=torch.long),
            'target': torch.tensor(sample['target'], dtype=torch.long),
            'genre': torch.tensor(sample['genre'], dtype=torch.long),
            'mood': torch.tensor(sample['mood'], dtype=torch.long)
        }


class MusicTrainer:
    """Trainer for music generation models"""

    def __init__(
        self,
        model: nn.Module,
        model_type: str = 'transformer',
        device: str = 'cpu',
        learning_rate: float = 0.001
    ):
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(
        self,
        dataloader: DataLoader,
        conditional: bool = False
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc='Training'):
            self.optimizer.zero_grad()

            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            # Forward pass
            if conditional:
                genres = batch['genre'].to(self.device)
                moods = batch['mood'].to(self.device)

                if self.model_type == 'lstm':
                    outputs, _ = self.model(inputs, genres, moods)
                else:  # transformer
                    outputs = self.model(inputs, genres, moods)
            else:
                if self.model_type == 'lstm':
                    outputs, _ = self.model(inputs)
                else:
                    outputs = self.model(inputs)

            # Calculate loss
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(
        self,
        dataloader: DataLoader,
        conditional: bool = False
    ) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                if conditional:
                    genres = batch['genre'].to(self.device)
                    moods = batch['mood'].to(self.device)

                    if self.model_type == 'lstm':
                        outputs, _ = self.model(inputs, genres, moods)
                    else:
                        outputs = self.model(inputs, genres, moods)
                else:
                    if self.model_type == 'lstm':
                        outputs, _ = self.model(inputs)
                    else:
                        outputs = self.model(inputs)

                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        conditional: bool = False,
        save_dir: str = 'checkpoints',
        early_stopping_patience: int = 10
    ):
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            conditional: Whether model is conditional
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, conditional)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader, conditional)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                print(f"Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                self.save_checkpoint(checkpoint_path, epoch, val_loss)

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        return checkpoint['epoch'], checkpoint['val_loss']


class DataPreprocessor:
    """Preprocess MIDI files for training"""

    def __init__(self):
        from ..utils.midi_processor import MIDIProcessor
        self.midi_processor = MIDIProcessor()

    def load_midi_files(
        self,
        file_paths: List[str],
        max_length: Optional[int] = None
    ) -> List[np.ndarray]:
        """Load multiple MIDI files"""
        sequences = []

        for file_path in file_paths:
            try:
                notes, _, _ = self.midi_processor.midi_to_notes(file_path)

                if max_length:
                    notes = notes[:max_length]

                if len(notes) > 0:
                    sequences.append(notes)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return sequences

    def prepare_dataset(
        self,
        sequences: List[np.ndarray],
        seq_length: int = 50,
        train_split: float = 0.8,
        genres: Optional[List[int]] = None,
        moods: Optional[List[int]] = None
    ) -> Tuple[MusicDataset, MusicDataset]:
        """Prepare train and validation datasets"""

        # Split data
        split_idx = int(len(sequences) * train_split)
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]

        if genres:
            train_genres = genres[:split_idx]
            val_genres = genres[split_idx:]
        else:
            train_genres = None
            val_genres = None

        if moods:
            train_moods = moods[:split_idx]
            val_moods = moods[split_idx:]
        else:
            train_moods = None
            val_moods = None

        # Create datasets
        train_dataset = MusicDataset(train_sequences, seq_length, train_genres, train_moods)
        val_dataset = MusicDataset(val_sequences, seq_length, val_genres, val_moods)

        return train_dataset, val_dataset

    def augment_data(
        self,
        sequences: List[np.ndarray],
        num_augmentations: int = 3
    ) -> List[np.ndarray]:
        """Augment data with transpositions"""
        from ..utils.midi_processor import MIDIAugmenter

        augmented = list(sequences)  # Include originals

        for seq in sequences:
            for _ in range(num_augmentations):
                # Random transpose
                transposed = MIDIAugmenter.transpose(
                    seq,
                    np.random.randint(-5, 6)
                )
                augmented.append(transposed)

        return augmented
