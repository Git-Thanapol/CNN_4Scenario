import numpy as np
import librosa
from typing import List
from .config import SAMPLE_RATE, DURATION, WINDOW_SIZE, STRIDE, N_MELS, HOP_LENGTH

def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Loads audio file. (Mock implementation returns random noise)."""
    # Real implementation: y, _ = librosa.load(file_path, sr=sr)
    # For demo, returning random noise of length ~4s (+- variation)
    length = int(sr * (DURATION + np.random.uniform(-0.5, 0.5)))
    return np.random.randn(length).astype(np.float32)

def denoise_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Applies noise reduction (Placeholder for noisereduce library)."""
    # import noisereduce as nr
    # return nr.reduce_noise(y=y, sr=sr)
    return y * 0.9  # Mock effect

def segment_audio(y: np.ndarray, sr: int, window_size: float, stride: float) -> List[np.ndarray]:
    """
    Slices audio into segments with padding for edge cases.
    CRITICAL: Handles the last segment padding (e.g. 3.5s - 4.5s).
    """
    window_samples = int(window_size * sr)
    stride_samples = int(stride * sr)
    segments = []
    
    total_samples = len(y)
    
    # Sliding window logic
    for start in range(0, total_samples, stride_samples):
        end = start + window_samples
        
        # Stop if the start is beyond the signal (shouldn't happen with loop logic but safe guard)
        if start >= total_samples:
            break
            
        # Check if we have enough data, if not -> Pad
        if end > total_samples:
            # Calculate needed padding
            pad_width = end - total_samples
            # Extract what we have
            segment = y[start:]
            # Pad with zeros at the end
            segment = np.pad(segment, (0, pad_width), mode='constant')
        else:
            segment = y[start:end]
            
        segments.append(segment)
        
    return segments

def compute_spectrogram(y: np.ndarray, sr: int, method: str = 'logmel') -> np.ndarray:
    """Computes spectrogram based on method."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    
    if method == 'pcen':
        # PCEN: Great for non-stationary noise
        return librosa.pcen(S * (2**31), sr=sr, hop_length=HOP_LENGTH)
    else:
        # Default: Log-Mel
        return librosa.power_to_db(S, ref=np.max)
