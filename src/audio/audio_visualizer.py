"""
Audio Visualizer
Real-time audio visualization and monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
from typing import Optional
from .audio_config import AudioConfig


class AudioVisualizer:
    """
    Real-time audio visualization
    Displays waveform, spectrum, and levels
    """

    def __init__(self, config: AudioConfig):
        """
        Initialize visualizer

        Args:
            config: AudioConfig object
        """
        self.config = config
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_running = False

        # Visualization parameters
        self.window_duration = 2.0  # seconds
        self.window_samples = int(self.window_duration * config.sample_rate)

        # Data buffers
        self.waveform_buffer = np.zeros(self.window_samples)
        self.spectrum_buffer = np.zeros(config.n_fft // 2 + 1)

        # Matplotlib setup
        self.fig = None
        self.axes = None
        self.lines = None

    def update_audio(self, audio_data: np.ndarray):
        """
        Update visualizer with new audio data

        Args:
            audio_data: Audio chunk
        """
        try:
            self.audio_queue.put_nowait(audio_data.flatten())
        except queue.Full:
            # Drop oldest data
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data.flatten())
            except:
                pass

    def _init_plot(self):
        """Initialize matplotlib plot"""
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('Real-Time Audio Monitoring', fontsize=14, fontweight='bold')

        # Waveform plot
        time_axis = np.arange(self.window_samples) / self.config.sample_rate
        self.lines = []

        # Waveform
        ax1 = self.axes[0]
        line1, = ax1.plot(time_axis, self.waveform_buffer, 'b-', linewidth=0.5)
        ax1.set_ylim(-1.0, 1.0)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Waveform')
        ax1.grid(True, alpha=0.3)
        self.lines.append(line1)

        # Spectrum
        ax2 = self.axes[1]
        freq_axis = np.fft.rfftfreq(self.config.n_fft, 1/self.config.sample_rate)
        line2, = ax2.plot(freq_axis, self.spectrum_buffer, 'r-', linewidth=1)
        ax2.set_xlim(0, self.config.sample_rate / 2)
        ax2.set_ylim(-80, 0)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title('Frequency Spectrum')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        self.lines.append(line2)

        # Level meter
        ax3 = self.axes[2]
        line3 = ax3.barh([0], [0], color='green')
        ax3.set_xlim(-60, 0)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_xlabel('Level (dB)')
        ax3.set_title('Audio Level (RMS)')
        ax3.set_yticks([])
        ax3.grid(True, alpha=0.3, axis='x')
        self.lines.append(line3)

        plt.tight_layout()

    def _update_plot(self, frame):
        """Update plot with new data"""
        try:
            # Get audio data from queue
            while not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get_nowait()

                # Update waveform buffer
                chunk_len = len(audio_chunk)
                if chunk_len > 0:
                    self.waveform_buffer = np.roll(self.waveform_buffer, -chunk_len)
                    self.waveform_buffer[-chunk_len:] = audio_chunk

            # Update waveform plot
            self.lines[0].set_ydata(self.waveform_buffer)

            # Compute and update spectrum
            if len(self.waveform_buffer) > 0:
                spectrum = np.fft.rfft(self.waveform_buffer * np.hanning(len(self.waveform_buffer)))
                magnitude = np.abs(spectrum)
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))

                # Smooth spectrum
                self.spectrum_buffer = 0.7 * self.spectrum_buffer + 0.3 * magnitude_db
                self.lines[1].set_ydata(self.spectrum_buffer)

                # Update level meter
                rms = np.sqrt(np.mean(self.waveform_buffer**2))
                rms_db = 20 * np.log10(max(rms, 1e-10))

                # Update bar
                self.lines[2][0].set_width(rms_db)

                # Color code based on level
                if rms_db > -6:
                    self.lines[2][0].set_color('red')
                elif rms_db > -12:
                    self.lines[2][0].set_color('yellow')
                else:
                    self.lines[2][0].set_color('green')

        except Exception as e:
            print(f"Visualization update error: {e}")

        return self.lines

    def start(self, blocking: bool = False):
        """
        Start visualization

        Args:
            blocking: Whether to block until window is closed
        """
        if self.is_running:
            return

        self.is_running = True
        self._init_plot()

        # Create animation
        anim = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=50,  # 50ms update rate (20 FPS)
            blit=False,
            cache_frame_data=False
        )

        if blocking:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.001)

    def stop(self):
        """Stop visualization"""
        if not self.is_running:
            return

        self.is_running = False
        if self.fig:
            plt.close(self.fig)


class ConsoleVisualizer:
    """
    Simple console-based audio level meter
    For environments without GUI support
    """

    def __init__(self, width: int = 50):
        """
        Initialize console visualizer

        Args:
            width: Width of level meter in characters
        """
        self.width = width

    def display_level(self, level_db: float):
        """
        Display audio level as console meter

        Args:
            level_db: Level in dB
        """
        # Map dB to meter width (-60 dB to 0 dB)
        normalized = (level_db + 60) / 60
        normalized = max(0.0, min(1.0, normalized))

        filled = int(normalized * self.width)
        empty = self.width - filled

        # Color coding
        if level_db > -6:
            bar_char = '█'
            color = 'RED'
        elif level_db > -12:
            bar_char = '▓'
            color = 'YEL'
        else:
            bar_char = '▒'
            color = 'GRN'

        # Build meter
        meter = f"[{bar_char * filled}{' ' * empty}]"

        # Display
        print(f"\r{color} {meter} {level_db:+6.1f} dB", end='', flush=True)

    def display_spectrum_summary(self, audio: np.ndarray, sample_rate: int):
        """
        Display simplified spectrum summary

        Args:
            audio: Audio data
            sample_rate: Sample rate
        """
        # Compute FFT
        spectrum = np.fft.rfft(audio)
        magnitude = np.abs(spectrum)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

        # Divide into frequency bands
        bands = [
            ('Low', 20, 250),
            ('Mid', 250, 2000),
            ('High', 2000, 8000),
            ('VHigh', 8000, sample_rate/2)
        ]

        print("\n\nFrequency Bands:")
        for name, f_low, f_high in bands:
            mask = (freqs >= f_low) & (freqs < f_high)
            band_magnitude = np.mean(magnitude[mask])
            band_db = 20 * np.log10(max(band_magnitude, 1e-10))

            # Simple bar
            normalized = (band_db + 60) / 60
            normalized = max(0.0, min(1.0, normalized))
            bar_len = int(normalized * 20)

            print(f"  {name:6s} ({f_low:5.0f}-{f_high:5.0f} Hz): {'█' * bar_len} {band_db:+6.1f} dB")
