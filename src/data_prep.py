import glob
import os
import pandas as pd
import numpy as np
import logging
from .config import SAMPLE_RATE, WINDOW_SIZE, STRIDE, CLASSES, LOW_PASS_CUTOFF, HIGH_PASS_CUTOFF, AUGMENT_RAW_DATA
from .audio_processing import load_audio, denoise_audio_stationary, denoise_audio_nonstationary , segment_audio, compute_spectrogram, apply_low_pass_filter, apply_band_pass_filter
from .augmentation import RawAudioAugmentor
import random

logger = logging.getLogger(__name__)

def load_metadata(folder_path: str) -> pd.DataFrame:
    """
    Loads metadata from real WAV files in the specified folder.
    Expected pattern: Case_PWM_Iteration.wav matches one of the classes.
    """
    folder_path = os.path.expanduser(folder_path)
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    
    data = []
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        
        # Heuristic parsing based on "Case_PWM_Iteration"
        # Example: Healthy_1300_1.wav -> case=Healthy, pwm=1300, iter=1
        if len(parts) >= 3:
            raw_case = parts[0].lower()
            pwm = parts[1]
            iteration = parts[2]
            
            # Map IMBALANCE -> UNBALANCE
            if raw_case == 'imbalance':
                label = 'imbalance'
            elif raw_case == 'healthy':
                label = 'healthy'
            elif raw_case == 'fouling':
                label = 'fouling'
            elif raw_case == 'seaweed':
                label = 'seaweed'
            else:
                logger.warning(f"Unknown label in file: {filename}. Skipping.")
                continue
                
            data.append({
                "file_id": name_no_ext,
                "file_path": file_path,
                "label": label,
                "pwm": pwm,
                "iteration": iteration
            })
        else:
            logger.warning(f"Skipping malformed filename: {filename}")

    if not data:
        raise ValueError(f"No valid wav files found in {folder_path} matching pattern Case_PWM_Iter")

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} files from {folder_path}")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")
    return df

def prepare_data_for_fold(df_meta, train_idx, val_idx, experiment_type, augment=False):
    """
    Splits by FILE ID first, then processes audio -> segments -> features.
    This guarantees NO DATA LEAKAGE.
    augment: If True, applies raw audio augmentation to training data.
    """
    
    # Initialize Augmentor once
    raw_augmentor = RawAudioAugmentor(sr=SAMPLE_RATE)
    
    def process_subset(subset_ids, subset_labels, is_training=False):
        X_list, y_list = [], []
        
        for file_id, label in zip(subset_ids, subset_labels):
            # 1. Load Audio
            file_path = df_meta[df_meta['file_id'] == file_id]['file_path'].values[0]
            y_orig = load_audio(file_path)
            
            # Prepare list of audios to process (Original + Augmented)
            audios_to_process = [y_orig]
            
            # --- Augmentation Logic ---
            if is_training and augment and AUGMENT_RAW_DATA:
                # Create ONE augmented version per file to double the dataset size
                # Randomly select an augmentation strategy
                choice = random.choice(['time_shift', 'pitch_shift', 'gaussian_noise', 'pink_noise'])
                
                y_aug = y_orig.copy()
                if choice == 'time_shift':
                    y_aug = raw_augmentor.time_shift(y_aug)
                elif choice == 'pitch_shift':
                    y_aug = raw_augmentor.pitch_shift(y_aug, n_steps=random.choice([-2, -1, 1, 2]))
                elif choice == 'gaussian_noise':
                    y_aug = raw_augmentor.add_gaussian_noise_snr(y_aug, snr_db=random.uniform(10, 20))
                elif choice == 'pink_noise':
                    y_aug = raw_augmentor.add_pink_noise_snr(y_aug, snr_db=random.uniform(10, 20))
                
                audios_to_process.append(y_aug)
            # --------------------------
            
            for y in audios_to_process:
                # 1.5 Apply Band Pass Filter (Global for all experiments)
                y = apply_band_pass_filter(y, SAMPLE_RATE, HIGH_PASS_CUTOFF, LOW_PASS_CUTOFF)
                
                # 2. Denoising (Based on Experiment)
                if experiment_type in ['Proposed_1_Denoised_Stationary_LogMel', 'Proposed_4_Mix']:
                    y = denoise_audio_stationary(y, SAMPLE_RATE)
                # Note: For 'Proposed_3_Mix', you'd ideally mix raw and denoised here
                # But for simplicity, let's treat it as a separate augmentation logic if needed
                if experiment_type in ['Proposed_2_Denoised_Nonstationary_LogMel']:
                    y = denoise_audio_nonstationary(y, SAMPLE_RATE)

                # 3. Segmentation (Sliding Window + Padding)
                segments = segment_audio(y, SAMPLE_RATE, WINDOW_SIZE, STRIDE)
                
                # 4. Feature Extraction (LogMel vs PCEN)
                feature_method = 'pcen' if 'PCEN' in experiment_type else 'logmel'
                
                for seg in segments:
                    spec = compute_spectrogram(seg, SAMPLE_RATE, method=feature_method)
                    X_list.append(spec)
                    y_list.append(CLASSES.index(label)) # All segments inherit the file's label as int index
                
        return np.array(X_list), np.array(y_list)

    # Split IDs
    train_ids, train_labels_id = df_meta.iloc[train_idx]['file_id'], df_meta.iloc[train_idx]['label']
    val_ids, val_labels_id = df_meta.iloc[val_idx]['file_id'], df_meta.iloc[val_idx]['label']
    
    # Process Audio
    logger.info(f"Processing Train Set for {experiment_type}...")
    X_train, y_train = process_subset(train_ids, train_labels_id, is_training=True)
    logger.info(f"Processing Val Set for {experiment_type}...")
    X_val, y_val = process_subset(val_ids, val_labels_id, is_training=False)
    
    return X_train, y_train, X_val, y_val
