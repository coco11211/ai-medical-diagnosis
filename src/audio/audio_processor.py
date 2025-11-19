"""
Real-Time Audio Processor
Integrates audio capture and noise cancellation for live processing
"""

import numpy as np
import sounddevice as sd
import threading
import time
from pathlib import Path
from typing import Optional, Callable
import wave
from .audio_capture import AudioCapture
from .noise_canceller import NoiseCanceller
from .audio_config import AudioConfig


class AudioProcessor:
    """
    Real-time audio processing pipeline
    Captures audio, applies noise cancellation, and outputs cleaned audio
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio processor

        Args:
            config: AudioConfig object (creates default if None)
        """
        self.config = config or AudioConfig()
        self.capture = AudioCapture(self.config)
        self.canceller = NoiseCanceller(self.config)

        self.is_processing = False
        self.processing_thread = None
        self.output_stream = None

        # Statistics
        self.frames_processed = 0
        self.total_latency = 0.0
        self.processing_times = []

        # Output recording
        self.recorded_frames = []
        self.record_output = False

    def list_devices(self):
        """List available audio devices"""
        print("\n=== Available Audio Devices ===")
        devices = self.capture.list_devices()

        for device in devices:
            print(f"  [{device['id']}] {device['name']}")
            print(f"      Channels: {device['channels']}, Sample Rate: {device['sample_rate']} Hz")

        print("\n=== Current Default Device ===")
        default = self.capture.get_default_device()
        print(f"  [{default['id']}] {default['name']}")
        print()

    def capture_noise_profile(self, duration: Optional[float] = None):
        """
        Capture noise profile for adaptive noise cancellation

        Args:
            duration: Duration in seconds
        """
        noise_audio = self.capture.capture_noise_profile(duration)
        self.canceller.set_noise_profile(noise_audio)

    def _processing_loop(self):
        """Main processing loop (runs in separate thread)"""
        print("\nStarting real-time audio processing...")
        print("Press Ctrl+C to stop\n")

        while self.is_processing:
            try:
                # Get audio chunk from capture
                audio_chunk = self.capture.get_audio_chunk(timeout=0.1)

                if audio_chunk is None:
                    continue

                # Measure processing time
                start_time = time.perf_counter()

                # Apply noise cancellation
                cleaned_chunk = self.canceller.process_chunk(audio_chunk)

                # Calculate processing time
                processing_time = time.perf_counter() - start_time
                self.processing_times.append(processing_time)

                # Keep only last 100 measurements
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)

                # Output cleaned audio
                if self.output_stream and self.config.monitor_enabled:
                    try:
                        # Reshape for output
                        output_data = cleaned_chunk.reshape(-1, self.config.channels)
                        self.output_stream.write(output_data.astype(np.float32))
                    except Exception as e:
                        print(f"Output error: {e}")

                # Record if enabled
                if self.record_output:
                    self.recorded_frames.append(cleaned_chunk)

                # Update statistics
                self.frames_processed += 1
                self.total_latency += processing_time

                # Print statistics periodically
                if self.frames_processed % 100 == 0:
                    self._print_statistics()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Processing error: {e}")
                continue

    def _print_statistics(self):
        """Print processing statistics"""
        if len(self.processing_times) == 0:
            return

        avg_latency = np.mean(self.processing_times) * 1000  # ms
        max_latency = np.max(self.processing_times) * 1000  # ms
        min_latency = np.min(self.processing_times) * 1000  # ms

        print(f"\r[Stats] Frames: {self.frames_processed:6d} | "
              f"Latency: {avg_latency:5.2f}ms (min: {min_latency:5.2f}ms, max: {max_latency:5.2f}ms) | "
              f"RT Factor: {(avg_latency / (self.config.chunk_size / self.config.sample_rate * 1000)):.2f}x",
              end='', flush=True)

    def start_processing(self, monitor: bool = True, record: bool = False):
        """
        Start real-time audio processing

        Args:
            monitor: Enable audio monitoring (playback cleaned audio)
            record: Record cleaned audio to file
        """
        if self.is_processing:
            print("Processing already running")
            return

        self.config.monitor_enabled = monitor
        self.record_output = record

        # Start audio capture
        self.capture.start_capture()

        # Start output stream if monitoring enabled
        if monitor:
            try:
                self.output_stream = sd.OutputStream(
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    device=self.config.output_device,
                    dtype=np.float32
                )
                self.output_stream.start()
                print("Audio monitoring enabled")
            except Exception as e:
                print(f"Warning: Could not start output stream: {e}")
                self.output_stream = None

        # Start processing thread
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def stop_processing(self):
        """Stop audio processing"""
        if not self.is_processing:
            return

        print("\n\nStopping audio processing...")

        # Stop processing loop
        self.is_processing = False

        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        # Stop capture
        self.capture.stop_capture()

        # Stop output stream
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None

        # Save recorded audio if enabled
        if self.record_output and len(self.recorded_frames) > 0:
            self._save_recorded_audio()

        # Print final statistics
        print("\n\n=== Final Statistics ===")
        print(f"Total frames processed: {self.frames_processed}")
        if self.frames_processed > 0:
            avg_latency = self.total_latency / self.frames_processed * 1000
            print(f"Average latency: {avg_latency:.2f} ms")
        print()

    def _save_recorded_audio(self):
        """Save recorded audio to WAV file"""
        try:
            # Create output directory
            output_dir = Path(self.config.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.output_path}_{timestamp}.wav"

            # Concatenate all recorded frames
            audio_data = np.concatenate(self.recorded_frames)

            # Normalize to int16 range
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)

            # Save to WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_data.tobytes())

            print(f"Recorded audio saved to: {filename}")

        except Exception as e:
            print(f"Error saving recorded audio: {e}")

    def process_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Process an audio file with noise cancellation

        Args:
            input_file: Path to input audio file
            output_file: Path to output file (auto-generated if None)

        Returns:
            Path to output file
        """
        print(f"Processing file: {input_file}")

        try:
            # Read input file
            with wave.open(input_file, 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)

            # Convert to float
            audio_data = audio_data.astype(np.float32) / 32768.0

            # Process in chunks
            chunk_size = self.config.chunk_size
            processed_chunks = []

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]

                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                # Process chunk
                cleaned_chunk = self.canceller.process_chunk(chunk)
                processed_chunks.append(cleaned_chunk)

            # Concatenate processed chunks
            processed_audio = np.concatenate(processed_chunks)

            # Trim to original length
            processed_audio = processed_audio[:len(audio_data)]

            # Convert back to int16
            processed_audio = np.clip(processed_audio, -1.0, 1.0)
            processed_audio = (processed_audio * 32767).astype(np.int16)

            # Generate output filename
            if output_file is None:
                input_path = Path(input_file)
                output_file = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")

            # Save output file
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(processed_audio.tobytes())

            print(f"Processed file saved to: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error processing file: {e}")
            raise

    def get_statistics(self) -> dict:
        """Get processing statistics"""
        if len(self.processing_times) == 0:
            return {}

        return {
            'frames_processed': self.frames_processed,
            'average_latency_ms': np.mean(self.processing_times) * 1000,
            'min_latency_ms': np.min(self.processing_times) * 1000,
            'max_latency_ms': np.max(self.processing_times) * 1000,
            'realtime_factor': (np.mean(self.processing_times) /
                               (self.config.chunk_size / self.config.sample_rate))
        }

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_processing()
