"""
Audio Capture Module
Real-time audio input capture optimized for Windows 11
"""

import numpy as np
import sounddevice as sd
import queue
import threading
from typing import Optional, Callable, List
from .audio_config import AudioConfig


class AudioCapture:
    """
    Real-time audio capture with support for Windows WASAPI
    Optimized for low-latency audio processing on Windows 11
    """

    def __init__(self, config: AudioConfig):
        """
        Initialize audio capture

        Args:
            config: AudioConfig object with capture settings
        """
        self.config = config
        self.audio_queue = queue.Queue(maxsize=config.buffer_size)
        self.is_capturing = False
        self.stream = None
        self.callbacks: List[Callable] = []
        self.noise_profile = None
        self.noise_profile_frames = []

        # Configure sounddevice for Windows 11
        if config.use_wasapi:
            sd.default.hostapi = self._get_wasapi_hostapi()

    def _get_wasapi_hostapi(self) -> Optional[int]:
        """Get WASAPI host API index for Windows"""
        try:
            hostapis = sd.query_hostapis()
            for i, api in enumerate(hostapis):
                if 'WASAPI' in api['name']:
                    return i
        except Exception as e:
            print(f"Warning: Could not find WASAPI, using default: {e}")
        return None

    def list_devices(self) -> List[dict]:
        """List available audio input devices"""
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })

        return input_devices

    def get_default_device(self) -> dict:
        """Get default input device information"""
        device_id = sd.default.device[0]  # Input device
        device_info = sd.query_devices(device_id)
        return {
            'id': device_id,
            'name': device_info['name'],
            'channels': device_info['max_input_channels'],
            'sample_rate': device_info['default_samplerate']
        }

    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio stream status: {status}")

        # Convert to numpy array and copy
        audio_data = indata.copy()

        # Add to queue for processing
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            # Drop oldest frame if queue is full
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data)
            except:
                pass

        # Call registered callbacks
        for callback in self.callbacks:
            try:
                callback(audio_data, time)
            except Exception as e:
                print(f"Callback error: {e}")

    def register_callback(self, callback: Callable):
        """
        Register a callback function to be called on each audio frame

        Args:
            callback: Function with signature callback(audio_data, time_info)
        """
        self.callbacks.append(callback)

    def capture_noise_profile(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Capture noise profile for adaptive noise cancellation

        Args:
            duration: Duration in seconds (uses config if None)

        Returns:
            Noise profile as numpy array
        """
        if duration is None:
            duration = self.config.noise_profile_duration

        print(f"Capturing noise profile for {duration} seconds...")
        print("Please remain silent or maintain ambient noise...")

        self.noise_profile_frames = []

        def noise_callback(audio_data, time_info):
            self.noise_profile_frames.append(audio_data.flatten())

        # Temporarily add noise capture callback
        self.register_callback(noise_callback)

        # Capture for specified duration
        frames_needed = int(duration * self.config.sample_rate / self.config.chunk_size)

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_size,
            device=self.config.input_device,
            callback=lambda indata, frames, time, status: noise_callback(indata, time)
        ):
            while len(self.noise_profile_frames) < frames_needed:
                sd.sleep(100)

        # Remove noise callback
        self.callbacks.remove(noise_callback)

        # Compute noise profile (average spectrum)
        self.noise_profile = np.concatenate(self.noise_profile_frames)
        print("Noise profile captured successfully!")

        return self.noise_profile

    def start_capture(self) -> None:
        """Start real-time audio capture"""
        if self.is_capturing:
            print("Warning: Capture already running")
            return

        # Apply latency settings
        latency_settings = self.config.get_latency_settings()
        self.config.chunk_size = latency_settings['chunk_size']
        self.config.buffer_size = latency_settings['buffer_size']

        # Create input stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                device=self.config.input_device,
                callback=self._audio_callback,
                dtype=np.float32
            )

            self.stream.start()
            self.is_capturing = True
            print(f"Audio capture started:")
            print(f"  Device: {self.get_default_device()['name']}")
            print(f"  Sample rate: {self.config.sample_rate} Hz")
            print(f"  Chunk size: {self.config.chunk_size}")
            print(f"  Latency mode: {self.config.latency_mode}")

        except Exception as e:
            print(f"Error starting audio capture: {e}")
            raise

    def stop_capture(self) -> None:
        """Stop audio capture"""
        if not self.is_capturing:
            return

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.is_capturing = False
        print("Audio capture stopped")

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next audio chunk from queue

        Args:
            timeout: Maximum time to wait for chunk

        Returns:
            Audio chunk as numpy array or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_audio_level(self) -> float:
        """
        Get current audio input level (RMS)

        Returns:
            RMS level in dB
        """
        try:
            chunk = self.audio_queue.get_nowait()
            rms = np.sqrt(np.mean(chunk**2))
            db = 20 * np.log10(max(rms, 1e-10))
            # Put chunk back
            self.audio_queue.put_nowait(chunk)
            return db
        except queue.Empty:
            return -np.inf

    def __enter__(self):
        """Context manager entry"""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_capture()
