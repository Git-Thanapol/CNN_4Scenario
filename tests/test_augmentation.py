import numpy as np
import librosa
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.augmentation import RawAudioAugmentor, SpectrogramAugmentor
from src.config import SAMPLE_RATE

def test_raw_audio_augmentation():
    print("--- Testing Raw Audio Augmentation ---")
    sr = SAMPLE_RATE
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t) # 440 Hz Sine Wave
    
    augmentor = RawAudioAugmentor(sr=sr)
    
    # 1. Time Shift
    y_shifted = augmentor.time_shift(y, shift_max=0.5)
    assert len(y_shifted) == len(y), "Time shift should preserve length"
    print("Time Shift: PASS")
    
    # 2. Pitch Shift
    y_pitch = augmentor.pitch_shift(y, n_steps=4)
    assert len(y_pitch) == len(y), "Pitch shift should preserve length"
    print("Pitch Shift: PASS")
    
    # 3. Time Stretch
    y_stretch_fast = augmentor.time_stretch(y, rate=1.5)
    assert len(y_stretch_fast) < len(y), "Fast stretch should shorten audio"
    
    y_stretch_slow = augmentor.time_stretch(y, rate=0.5)
    assert len(y_stretch_slow) > len(y), "Slow stretch should lengthen audio"
    print("Time Stretch: PASS")
    
    # 4. Gaussian Noise
    y_noise = augmentor.add_gaussian_noise(y, amplitude=0.1)
    assert len(y_noise) == len(y)
    assert np.std(y_noise) > np.std(y), "Noise should increase variance (mostly)"
    print("Gaussian Noise: PASS")
    
    # 5. Gaussian Noise SNR
    y_snr = augmentor.add_gaussian_noise_snr(y, snr_db=10)
    assert len(y_snr) == len(y)
    print("Gaussian Noise SNR: PASS")
    
    # 6. Pink Noise SNR
    y_pink = augmentor.add_pink_noise_snr(y, snr_db=10)
    assert len(y_pink) == len(y)
    print("Pink Noise SNR: PASS")

def test_spectrogram_augmentation():
    print("\n--- Testing Spectrogram Augmentation ---")
    # Dummy Spectrogram: 128 Mels x 100 Time Steps
    spec = np.ones((128, 100)) 
    
    augmentor = SpectrogramAugmentor()
    
    # 1. Freq Mask
    masked_spec = augmentor.freq_mask(spec, max_width=20, num_masks=2)
    assert masked_spec.shape == spec.shape
    assert np.any(masked_spec == 0), "Freq mask should introduce zeros"
    print("Freq Mask: PASS")
    
    # 2. Time Mask
    masked_spec_t = augmentor.time_mask(spec, max_width=20, num_masks=2)
    assert masked_spec_t.shape == spec.shape
    assert np.any(masked_spec_t == 0), "Time mask should introduce zeros"
    print("Time Mask: PASS")
    
    # 3. Random Masking
    masked_random = augmentor.random_masking(spec)
    assert masked_random.shape == spec.shape
    print("Random Masking: PASS")

if __name__ == "__main__":
    try:
        test_raw_audio_augmentation()
        test_spectrogram_augmentation()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
