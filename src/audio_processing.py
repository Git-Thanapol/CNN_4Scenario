import numpy as np
import librosa
from typing import List
import noisereduce as nr
import scipy.signal
from scipy.signal import butter, lfilter
from .config import SAMPLE_RATE, DURATION, WINDOW_SIZE, STRIDE, N_MELS, HOP_LENGTH

def apply_low_pass_filter(y: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
    """Applies a low-pass ButterWorth filter."""
    if cutoff >= sr / 2:
        return y # Nyquist limit check
        
    order = 6
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y_filtered = lfilter(b, a, y)
    return y_filtered

def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Loads audio file using librosa."""
    try:
        y, _ = librosa.load(file_path, sr=sr)
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])

def denoise_audio_nonstationary(y: np.ndarray, sr: int) -> np.ndarray:
    """Applies noise reduction (Placeholder for noisereduce library)."""
    return nr.reduce_noise(y=y, sr=sr , stationary=False)

def denoise_audio_stationary(y: np.ndarray, sr: int) -> np.ndarray:
    """Applies noise reduction (Placeholder for noisereduce library)."""
    return nr.reduce_noise(y=y, sr=sr, stationary=True)

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
