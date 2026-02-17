import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_processing import apply_band_pass_filter
from src.config import SAMPLE_RATE, LOW_PASS_CUTOFF, HIGH_PASS_CUTOFF

def test_band_pass_filter():
    print(f"Testing Band-Pass Filter: {HIGH_PASS_CUTOFF} Hz - {LOW_PASS_CUTOFF} Hz at SR={SAMPLE_RATE}")
    
    # Generate White Noise
    duration = 1.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    noise = np.random.normal(0, 1, len(t))
    
    # Apply Filter
    filtered_noise = apply_band_pass_filter(noise, SAMPLE_RATE, HIGH_PASS_CUTOFF, LOW_PASS_CUTOFF)
    
    # Compute PSD
    freqs, psd_input = scipy.signal.welch(noise, SAMPLE_RATE)
    freqs, psd_output = scipy.signal.welch(filtered_noise, SAMPLE_RATE)
    
    # Check Attenuation
    # Frequencies to check: 
    # Below High Pass (e.g., 1000 Hz) - Should be attenuated
    # Inside Pass Band (e.g., 5000 Hz) - Should be preserved
    # Above Low Pass (11000 Hz) is close to Nyquist (11025 Hz), so we can't reliably test >11025 Hz in discrete signal.
    
    idx_1k = np.argmin(np.abs(freqs - 1000))
    idx_5k = np.argmin(np.abs(freqs - 5000))
    
    power_in_1k = psd_input[idx_1k]
    power_out_1k = psd_output[idx_1k]
    attenuation_1k = 10 * np.log10(power_out_1k / power_in_1k)
    
    power_in_5k = psd_input[idx_5k]
    power_out_5k = psd_output[idx_5k]
    attenuation_5k = 10 * np.log10(power_out_5k / power_in_5k)
    
    print(f"Attenuation at 1000 Hz (Stop Band): {attenuation_1k:.2f} dB")
    print(f"Attenuation at 5000 Hz (Pass Band): {attenuation_5k:.2f} dB")
    
    # Assertions
    if attenuation_1k > -3:
        print("FAIL: 1000 Hz is not sufficiently attenuated.")
    elif attenuation_5k < -3:
        print("FAIL: 5000 Hz is too attenuated.")
    else:
        print("PASS: Filter characteristics match expectations (Nyquist limited).")

if __name__ == "__main__":
    test_band_pass_filter()
