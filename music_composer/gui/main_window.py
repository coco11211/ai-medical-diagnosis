"""
Main GUI application for AI Music Composer
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import torch
import numpy as np
from typing import Optional

# Import backend modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.lstm_model import ConditionalLSTM
from models.transformer_model import ConditionalTransformer
from utils.composition_engine import CompositionEngine, MoodBasedComposer
from export.audio_export import AudioExporter, AudioPlayer
from training.trainer import MusicTrainer, DataPreprocessor, MusicDataset
from torch.utils.data import DataLoader


class MusicComposerGUI:
    """Main GUI application"""

    def __init__(self, root):
        self.root = root
        self.root.title("AI Music Composer")
        self.root.geometry("900x700")

        # Backend components
        self.model = None
        self.engine = None
        self.composer = None
        self.exporter = AudioExporter()
        self.player = AudioPlayer()
        self.trainer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # State
        self.current_composition = None
        self.is_generating = False
        self.is_training = False

        # Initialize default model
        self._init_default_model()

        # Setup GUI
        self._setup_gui()

    def _init_default_model(self):
        """Initialize default Transformer model"""
        self.model = ConditionalTransformer(
            vocab_size=128,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1
        ).to(self.device)

        self.engine = CompositionEngine(self.model, 'transformer', self.device)
        self.composer = MoodBasedComposer(self.engine)

    def _setup_gui(self):
        """Setup GUI components"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.compose_tab = ttk.Frame(self.notebook)
        self.train_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.compose_tab, text="Compose")
        self.notebook.add(self.train_tab, text="Train Model")
        self.notebook.add(self.settings_tab, text="Settings")

        # Setup each tab
        self._setup_compose_tab()
        self._setup_train_tab()
        self._setup_settings_tab()

        # Status bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_compose_tab(self):
        """Setup composition tab"""
        # Main frame
        main_frame = ttk.Frame(self.compose_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Composition Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Genre selection
        ttk.Label(params_frame, text="Genre:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.genre_var = tk.StringVar(value='pop')
        genre_combo = ttk.Combobox(
            params_frame,
            textvariable=self.genre_var,
            values=['classical', 'jazz', 'pop', 'rock', 'electronic', 'blues', 'ambient', 'folk'],
            state='readonly',
            width=20
        )
        genre_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(5, 20))

        # Mood selection
        ttk.Label(params_frame, text="Mood:").grid(row=0, column=2, sticky=tk.W, pady=5)
        self.mood_var = tk.StringVar(value='happy')
        mood_combo = ttk.Combobox(
            params_frame,
            textvariable=self.mood_var,
            values=['happy', 'sad', 'energetic', 'calm', 'mysterious', 'romantic', 'aggressive', 'peaceful'],
            state='readonly',
            width=20
        )
        mood_combo.grid(row=0, column=3, sticky=tk.W, pady=5, padx=5)

        # Duration
        ttk.Label(params_frame, text="Duration (seconds):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.duration_var = tk.IntVar(value=30)
        duration_spin = ttk.Spinbox(
            params_frame,
            from_=10,
            to=300,
            textvariable=self.duration_var,
            width=10
        )
        duration_spin.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(5, 20))

        # Tempo
        ttk.Label(params_frame, text="Tempo (BPM):").grid(row=1, column=2, sticky=tk.W, pady=5)
        self.tempo_var = tk.IntVar(value=120)
        tempo_spin = ttk.Spinbox(
            params_frame,
            from_=40,
            to=200,
            textvariable=self.tempo_var,
            width=10
        )
        tempo_spin.grid(row=1, column=3, sticky=tk.W, pady=5, padx=5)

        # Temperature
        ttk.Label(params_frame, text="Creativity:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.temperature_var = tk.DoubleVar(value=1.0)
        temperature_scale = ttk.Scale(
            params_frame,
            from_=0.5,
            to=1.5,
            variable=self.temperature_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        temperature_scale.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(5, 20))
        ttk.Label(params_frame, textvariable=self.temperature_var).grid(row=2, column=2, sticky=tk.W, pady=5)

        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))

        self.add_harmony_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Add Harmony",
            variable=self.add_harmony_var
        ).grid(row=0, column=0, sticky=tk.W, padx=10)

        self.add_drums_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Add Drums",
            variable=self.add_drums_var
        ).grid(row=0, column=1, sticky=tk.W, padx=10)

        # Generation buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.generate_btn = ttk.Button(
            button_frame,
            text="Generate Music",
            command=self._generate_music,
            style='Accent.TButton'
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Stop Generation",
            command=self._stop_generation
        ).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(fill=tk.X, pady=(0, 10))

        # Playback frame
        playback_frame = ttk.LabelFrame(main_frame, text="Playback & Export", padding="10")
        playback_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            playback_frame,
            text="Play MIDI",
            command=self._play_midi
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            playback_frame,
            text="Stop",
            command=self._stop_playback
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            playback_frame,
            text="Export MIDI",
            command=self._export_midi
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            playback_frame,
            text="Export WAV",
            command=self._export_wav
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            playback_frame,
            text="Export MP3",
            command=self._export_mp3
        ).pack(side=tk.LEFT, padx=5)

        # Output log
        log_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(log_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)

    def _setup_train_tab(self):
        """Setup training tab"""
        main_frame = ttk.Frame(self.train_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Model Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_type_var = tk.StringVar(value='transformer')
        ttk.Radiobutton(
            model_frame,
            text="Transformer",
            variable=self.model_type_var,
            value='transformer'
        ).grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(
            model_frame,
            text="LSTM",
            variable=self.model_type_var,
            value='lstm'
        ).grid(row=0, column=2, sticky=tk.W, pady=5)

        # Data selection
        data_frame = ttk.LabelFrame(main_frame, text="Training Data", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            data_frame,
            text="Select MIDI Files",
            command=self._select_training_data
        ).pack(side=tk.LEFT, padx=5)

        self.data_label = ttk.Label(data_frame, text="No files selected")
        self.data_label.pack(side=tk.LEFT, padx=10)

        # Training parameters
        train_params_frame = ttk.LabelFrame(main_frame, text="Training Parameters", padding="10")
        train_params_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(train_params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(
            train_params_frame,
            from_=1,
            to=500,
            textvariable=self.epochs_var,
            width=10
        ).grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)

        ttk.Label(train_params_frame, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, pady=5)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(
            train_params_frame,
            from_=8,
            to=128,
            textvariable=self.batch_size_var,
            width=10
        ).grid(row=0, column=3, sticky=tk.W, pady=5, padx=5)

        ttk.Label(train_params_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(
            train_params_frame,
            textvariable=self.lr_var,
            width=10
        ).grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # Training buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.train_btn = ttk.Button(
            button_frame,
            text="Start Training",
            command=self._start_training
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Load Model",
            command=self._load_model
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Save Model",
            command=self._save_model
        ).pack(side=tk.LEFT, padx=5)

        # Training log
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.train_log = tk.Text(log_frame, height=15, wrap=tk.WORD)
        self.train_log.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(log_frame, command=self.train_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_log.config(yscrollcommand=scrollbar.set)

    def _setup_settings_tab(self):
        """Setup settings tab"""
        main_frame = ttk.Frame(self.settings_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Device settings
        device_frame = ttk.LabelFrame(main_frame, text="Device Settings", padding="10")
        device_frame.pack(fill=tk.X, pady=(0, 10))

        device_info = f"Current Device: {self.device.upper()}"
        if self.device == 'cuda':
            device_info += f" ({torch.cuda.get_device_name(0)})"

        ttk.Label(device_frame, text=device_info).pack(anchor=tk.W)

        # About
        about_frame = ttk.LabelFrame(main_frame, text="About", padding="10")
        about_frame.pack(fill=tk.X, pady=(0, 10))

        about_text = """AI Music Composer v1.0

A comprehensive music generation system using deep learning.

Features:
- LSTM and Transformer models
- Multi-genre support (Classical, Jazz, Pop, Rock, Electronic, Blues, Ambient, Folk)
- Mood-based composition
- Harmony and melody synthesis
- Real-time playback
- Export to MIDI, WAV, and MP3
- Custom model training

Powered by PyTorch"""

        ttk.Label(about_frame, text=about_text, justify=tk.LEFT).pack(anchor=tk.W)

    # Compose tab methods
    def _generate_music(self):
        """Generate music in background thread"""
        if self.is_generating:
            messagebox.showwarning("Warning", "Generation already in progress")
            return

        self.is_generating = True
        self.generate_btn.config(state='disabled')
        self.progress.start()

        thread = threading.Thread(target=self._generation_worker)
        thread.daemon = True
        thread.start()

    def _generation_worker(self):
        """Worker thread for music generation"""
        try:
            self._log("Starting music generation...")

            genre = self.genre_var.get()
            mood = self.mood_var.get()
            duration = self.duration_var.get()
            tempo = self.tempo_var.get()
            temperature = self.temperature_var.get()
            add_harmony = self.add_harmony_var.get()
            add_drums = self.add_drums_var.get()

            self._log(f"Genre: {genre}, Mood: {mood}, Duration: {duration}s, Tempo: {tempo} BPM")

            # Generate composition
            self.current_composition = self.composer.compose_for_mood(
                mood=mood,
                duration=duration,
                genre=genre,
                tempo=tempo
            )

            # Update composition with user preferences
            if not add_harmony:
                self.current_composition['tracks'] = [self.current_composition['tracks'][0]]

            self._log("Music generated successfully!")
            self._log(f"Generated {len(self.current_composition['melody'])} notes")

        except Exception as e:
            self._log(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Generation failed: {str(e)}")

        finally:
            self.is_generating = False
            self.root.after(0, lambda: self.generate_btn.config(state='normal'))
            self.root.after(0, self.progress.stop)

    def _stop_generation(self):
        """Stop music generation"""
        self.is_generating = False
        self._log("Generation stopped by user")

    def _play_midi(self):
        """Play generated MIDI"""
        if self.current_composition is None:
            messagebox.showwarning("Warning", "No composition to play. Generate music first.")
            return

        try:
            # Save temporary MIDI file
            temp_file = "temp_playback.mid"
            self.engine.save_composition(self.current_composition, temp_file)

            # Play
            self.player.play_midi(temp_file)
            self._log("Playing music...")

        except Exception as e:
            messagebox.showerror("Error", f"Playback failed: {str(e)}")

    def _stop_playback(self):
        """Stop playback"""
        self.player.stop()
        self._log("Playback stopped")

    def _export_midi(self):
        """Export to MIDI"""
        if self.current_composition is None:
            messagebox.showwarning("Warning", "No composition to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.engine.save_composition(self.current_composition, file_path)
                self._log(f"Exported to: {file_path}")
                messagebox.showinfo("Success", "MIDI file exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

    def _export_wav(self):
        """Export to WAV"""
        if self.current_composition is None:
            messagebox.showwarning("Warning", "No composition to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self._log("Converting to WAV...")
                temp_midi = "temp_export.mid"
                self.engine.save_composition(self.current_composition, temp_midi)
                self.exporter.midi_to_wav(temp_midi, file_path)

                if os.path.exists(temp_midi):
                    os.remove(temp_midi)

                self._log(f"Exported to: {file_path}")
                messagebox.showinfo("Success", "WAV file exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

    def _export_mp3(self):
        """Export to MP3"""
        if self.current_composition is None:
            messagebox.showwarning("Warning", "No composition to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self._log("Converting to MP3...")
                temp_midi = "temp_export.mid"
                self.engine.save_composition(self.current_composition, temp_midi)
                self.exporter.midi_to_mp3(temp_midi, file_path)

                if os.path.exists(temp_midi):
                    os.remove(temp_midi)

                self._log(f"Exported to: {file_path}")
                messagebox.showinfo("Success", "MP3 file exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

    # Training tab methods
    def _select_training_data(self):
        """Select MIDI files for training"""
        files = filedialog.askopenfilenames(
            title="Select MIDI Files",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")]
        )

        if files:
            self.training_files = list(files)
            self.data_label.config(text=f"{len(files)} files selected")
            self._train_log(f"Selected {len(files)} MIDI files")

    def _start_training(self):
        """Start model training"""
        if not hasattr(self, 'training_files') or not self.training_files:
            messagebox.showwarning("Warning", "Please select training data first")
            return

        if self.is_training:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        self.is_training = True
        self.train_btn.config(state='disabled')

        thread = threading.Thread(target=self._training_worker)
        thread.daemon = True
        thread.start()

    def _training_worker(self):
        """Worker thread for training"""
        try:
            self._train_log("Preparing training data...")

            # Load data
            preprocessor = DataPreprocessor()
            sequences = preprocessor.load_midi_files(self.training_files)
            self._train_log(f"Loaded {len(sequences)} sequences")

            # Augment data
            sequences = preprocessor.augment_data(sequences, num_augmentations=2)
            self._train_log(f"Augmented to {len(sequences)} sequences")

            # Prepare datasets
            train_dataset, val_dataset = preprocessor.prepare_dataset(
                sequences,
                seq_length=50,
                train_split=0.8
            )

            batch_size = self.batch_size_var.get()
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            self._train_log(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

            # Create model
            model_type = self.model_type_var.get()
            if model_type == 'transformer':
                model = ConditionalTransformer().to(self.device)
            else:
                model = ConditionalLSTM().to(self.device)

            self._train_log(f"Created {model_type} model")

            # Create trainer
            trainer = MusicTrainer(model, model_type, self.device, self.lr_var.get())

            # Train
            epochs = self.epochs_var.get()
            self._train_log(f"Starting training for {epochs} epochs...")

            trainer.train(
                train_loader,
                val_loader,
                num_epochs=epochs,
                conditional=True,
                save_dir='checkpoints'
            )

            self._train_log("Training completed!")

            # Update model
            self.model = model
            self.engine = CompositionEngine(model, model_type, self.device)
            self.composer = MoodBasedComposer(self.engine)

            messagebox.showinfo("Success", "Training completed successfully!")

        except Exception as e:
            self._train_log(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")

        finally:
            self.is_training = False
            self.root.after(0, lambda: self.train_btn.config(state='normal'))

    def _load_model(self):
        """Load saved model"""
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("PyTorch models", "*.pt *.pth"), ("All files", "*.*")]
        )

        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self._train_log(f"Model loaded from: {file_path}")
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def _save_model(self):
        """Save current model"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                }, file_path)
                self._train_log(f"Model saved to: {file_path}")
                messagebox.showinfo("Success", "Model saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    # Utility methods
    def _log(self, message: str):
        """Log to output text"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def _train_log(self, message: str):
        """Log to training text"""
        self.train_log.insert(tk.END, message + "\n")
        self.train_log.see(tk.END)


def main():
    """Main entry point"""
    root = tk.Tk()

    # Set theme
    style = ttk.Style()
    style.theme_use('clam')

    app = MusicComposerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
