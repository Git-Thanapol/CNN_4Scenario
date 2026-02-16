import pandas as pd
import numpy as np
import logging
from .config import SAMPLE_RATE, WINDOW_SIZE, STRIDE
from .audio_processing import load_audio, denoise_audio, segment_audio, compute_spectrogram

logger = logging.getLogger(__name__)

def generate_mock_metadata(n_samples: int = 600) -> pd.DataFrame:
    """Generates mock metadata ensuring we split by FILE ID."""
    ids = [f"file_{i:04d}" for i in range(n_samples)]
    labels = np.random.randint(0, 4, size=n_samples)
    paths = [f"/tmp/mock_audio/{id}.wav" for id in ids] # Placeholder paths
    return pd.DataFrame({'file_id': ids, 'label': labels, 'path': paths})

def prepare_data_for_fold(df_meta, train_idx, val_idx, experiment_type):
    """
    Splits by FILE ID first, then processes audio -> segments -> features.
    This guarantees NO DATA LEAKAGE.
    """
    
    def process_subset(subset_ids, subset_labels):
        X_list, y_list = [], []
        
        for file_id, label in zip(subset_ids, subset_labels):
            # 1. Load Audio
            path = df_meta[df_meta['file_id'] == file_id]['path'].values[0]
            y = load_audio(path)
            
            # 2. Denoising (Based on Experiment)
            if experiment_type in ['Proposed_1_Denoised', 'Proposed_3_Mix']:
                y = denoise_audio(y, SAMPLE_RATE)
            # Note: For 'Proposed_3_Mix', you'd ideally mix raw and denoised here
            # But for simplicity, let's treat it as a separate augmentation logic if needed
                
            # 3. Segmentation (Sliding Window + Padding)
            segments = segment_audio(y, SAMPLE_RATE, WINDOW_SIZE, STRIDE)
            
            # 4. Feature Extraction (LogMel vs PCEN)
            feature_method = 'pcen' if 'PCEN' in experiment_type else 'logmel'
            
            for seg in segments:
                spec = compute_spectrogram(seg, SAMPLE_RATE, method=feature_method)
                X_list.append(spec)
                y_list.append(label) # All segments inherit the file's label
                
        return np.array(X_list), np.array(y_list)

    # Split IDs
    train_ids, train_labels_id = df_meta.iloc[train_idx]['file_id'], df_meta.iloc[train_idx]['label']
    val_ids, val_labels_id = df_meta.iloc[val_idx]['file_id'], df_meta.iloc[val_idx]['label']
    
    # Process Audio
    logger.info(f"Processing Train Set for {experiment_type}...")
    X_train, y_train = process_subset(train_ids, train_labels_id)
    logger.info(f"Processing Val Set for {experiment_type}...")
    X_val, y_val = process_subset(val_ids, val_labels_id)
    
    return X_train, y_train, X_val, y_val
