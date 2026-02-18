import numpy as np
import librosa
import random
from typing import Optional, Union

class RawAudioAugmentor:
    """
    Augments raw audio waveforms.
    """
    def __init__(self, sr: int):
        self.sr = sr

    def time_shift(self, y: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """
        Shifts the audio in time by rolling the array.
        shift_max: Maximum shift fraction (e.g., 0.2 means shift by up to 20% of duration).
        """
        length = len(y)
        shift_samples = int(random.uniform(-shift_max, shift_max) * length)
        return np.roll(y, shift_samples)

    def pitch_shift(self, y: np.ndarray, n_steps: float = 2.0) -> np.ndarray:
        """
        Shifts the pitch of the waveform.
        n_steps: Number of semitones to shift.
        """
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

    def time_stretch(self, y: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """
        Stretches the time of the waveform.
        rate: > 1.0 speeds up, < 1.0 slows down.
        """
        # Note: This changes the length of the audio.
        return librosa.effects.time_stretch(y, rate=rate)

    def add_gaussian_noise(self, y: np.ndarray, amplitude: float = 0.005) -> np.ndarray:
        """
        Adds random Gaussian noise.
        """
        noise = np.random.randn(len(y)) * amplitude
        return y + noise

    def add_gaussian_noise_snr(self, y: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
        """
        Adds Gaussian noise at a specific Signal-to-Noise Ratio (SNR) in dB.
        """
        # Calculate signal power
        signal_power = np.mean(y ** 2)
        if signal_power == 0:
            return y
            
        # Calculate required noise power
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        return y + noise

    def add_pink_noise_snr(self, y: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
        """
        Adds Pink noise (1/f) at a specific SNR.
        """
        # 1. Generate Pink Noise using 1/f law approximation
        # Pink noise has power density inversely proportional to frequency
        n = len(y)
        uneven = n % 2
        X = np.random.randn(n // 2 + 1 + uneven) + 1j * np.random.randn(n // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.) # +1 to avoid division by zero
        y_pink = (np.fft.irfft(X / S)).real
        
        if uneven:
            y_pink = y_pink[:-1]
            
        # Normalize pink noise to unit power temporarily
        y_pink = y_pink / np.std(y_pink)
        
        # 2. Calculate signal power
        signal_power = np.mean(y ** 2)
        if signal_power == 0:
            return y
            
        # 3. Calculate target noise power
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # 4. Scale pink noise
        y_pink = y_pink * np.sqrt(noise_power)
        
        return y + y_pink


class SpectrogramAugmentor:
    """
    Augments spectrograms (Frequency and Time Masking).
    Based on SpecAugment.
    """
    def __init__(self):
        pass

    def freq_mask(self, spec: np.ndarray, max_width: int = 10, num_masks: int = 1) -> np.ndarray:
        """
        Applies frequency masking.
        spec: (n_mels, time_steps)
        """
        cloned = spec.copy()
        n_mels = cloned.shape[0]
        
        for _ in range(num_masks):
            f_width = random.randint(0, max_width)
            f_low = random.randint(0, n_mels - f_width)
            
            cloned[f_low : f_low + f_width, :] = 0
            # Alternatively, mask with mean, but 0 (silence) is standard for LogMel usually
            # Or min value of spectrogram
            
        return cloned

    def time_mask(self, spec: np.ndarray, max_width: int = 10, num_masks: int = 1) -> np.ndarray:
        """
        Applies time masking.
        spec: (n_mels, time_steps)
        """
        cloned = spec.copy()
        time_steps = cloned.shape[1]
        
        for _ in range(num_masks):
            t_width = random.randint(0, max_width)
            t_low = random.randint(0, time_steps - t_width)
            
            cloned[:, t_low : t_low + t_width] = 0
            
        return cloned

    def random_masking(self, spec: np.ndarray, 
                       freq_masks: int = 1, time_masks: int = 1,
                       freq_width: int = 20, time_width: int = 20) -> np.ndarray:
        """
        Applies both frequency and time masking.
        """
        spec = self.freq_mask(spec, max_width=freq_width, num_masks=freq_masks)
        spec = self.time_mask(spec, max_width=time_width, num_masks=time_masks)
        return spec
