import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from src.audio_processing import load_audio, apply_band_pass_filter, denoise_audio_stationary, denoise_audio_nonstationary, compute_spectrogram
from src.config import SAMPLE_RATE, LOW_PASS_CUTOFF, HIGH_PASS_CUTOFF

# File Path
file_path = r"sample_audio\HEALTHY_PWM1200_Iter5.wav"
output_dir = r"spectrogram_samples"
os.makedirs(output_dir, exist_ok=True)

def save_spectrogram(S, sr, title, filename):
    plt.figure(figsize=(10, 4))
    if 'PCEN' in title:
        librosa.display.specshow(S, sr=sr, hop_length=512, x_axis='time', y_axis='hz')
    else:
        librosa.display.specshow(S, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved {filename}")

def run():
    print(f"Processing {file_path}...")
    
    # 1. Load Audio
    y_raw = load_audio(file_path, sr=SAMPLE_RATE)
    if len(y_raw) == 0:
        print("Failed to load audio.")
        return

    # Take first 4 seconds for consistency if longer
    max_len = 4 * SAMPLE_RATE
    if len(y_raw) > max_len:
        y_raw = y_raw[:max_len]
        
    # 2. Apply Bandpass (Global Step)
    y_bandpass = apply_band_pass_filter(y_raw, SAMPLE_RATE, HIGH_PASS_CUTOFF, LOW_PASS_CUTOFF)
    
    # --- Experiment 1: Baseline LogMel ---
    # Just Bandpass -> LogMel
    spec_baseline = compute_spectrogram(y_bandpass, SAMPLE_RATE, method='logmel')
    save_spectrogram(spec_baseline, SAMPLE_RATE, "Baseline LogMel (Bandpass Only)", "spec_baseline.png")
    
    # --- Experiment 2: Stationary Denoising ---
    # Bandpass -> Stationary Denoise -> LogMel
    y_stat = denoise_audio_stationary(y_bandpass, SAMPLE_RATE)
    spec_stat = compute_spectrogram(y_stat, SAMPLE_RATE, method='logmel')
    save_spectrogram(spec_stat, SAMPLE_RATE, "Stationary Denoising LogMel", "spec_stationary.png")
    
    # --- Experiment 3: Non-Stationary Denoising ---
    # Bandpass -> Non-Stationary Denoise -> LogMel
    y_nonstat = denoise_audio_nonstationary(y_bandpass, SAMPLE_RATE)
    spec_nonstat = compute_spectrogram(y_nonstat, SAMPLE_RATE, method='logmel')
    save_spectrogram(spec_nonstat, SAMPLE_RATE, "Non-Stationary Denoising LogMel", "spec_nonstationary.png")
    
    # --- Experiment 4: Raw PCEN ---
    # Bandpass -> PCEN (No Denoising usually, or handled by PCEN)
    # Note: Experiment name is "Proposed_3_Raw_PCEN", implying raw audio?
    # But experiments_bandpass.py applies bandpass TO ALL.
    # Let's verify line 107 in data_prep.py: y = apply_band_pass_filter(...) happens for ALL.
    # So it is Bandpass -> PCEN.
    spec_pcen = compute_spectrogram(y_bandpass, SAMPLE_RATE, method='pcen')
    save_spectrogram(spec_pcen, SAMPLE_RATE, "PCEN (Bandpass Only)", "spec_pcen.png")

if __name__ == "__main__":
    run()
