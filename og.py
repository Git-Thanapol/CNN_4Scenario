import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.manifold import TSNE
import mlflow
import mlflow.pytorch
from typing import List, Tuple, Dict, Any

# --- Configuration ---
SAMPLE_RATE = 22050
DURATION = 4.0  # seconds
WINDOW_SIZE = 1.0  # seconds
STRIDE = 0.5  # seconds
N_MELS = 128
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 10  # Reduced for demo purposes
N_FOLDS = 5
CLASSES = ['Case1', 'Case2', 'Case3', 'Case4']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Mock Data Generation (For demonstration) ---
# In real usage, replace this with actual file loading logic
def generate_mock_metadata(n_samples: int = 600) -> pd.DataFrame:
    """Generates mock metadata ensuring we split by FILE ID."""
    ids = [f"file_{i:04d}" for i in range(n_samples)]
    labels = np.random.randint(0, 4, size=n_samples)
    paths = [f"/tmp/mock_audio/{id}.wav" for id in ids] # Placeholder paths
    return pd.DataFrame({'file_id': ids, 'label': labels, 'path': paths})

# --- Audio Processing & Feature Extraction ---

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

class AudioDataset(Dataset):
    """PyTorch Dataset handling pre-computed features."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1) # Add channel dim
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- Model Definition ---

class SimpleCNN(nn.Module):
    """
    A lightweight CNN for spectrogram classification.
    Returns features in forward() for t-SNE visualization.
    """
    def __init__(self, n_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        features = x.view(x.size(0), -1) # Flatten: [Batch, 64]
        out = self.fc(features)
        return out, features

# --- Evaluation & Plotting Helpers ---

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    filename = "confusion_matrix.png"
    plt.savefig(filename)
    plt.close()
    return filename

def plot_tsne(features, labels, classes):
    """Plots t-SNE of the extracted features."""
    if len(features) > 1000: # Subsample for speed if needed
        idx = np.random.choice(len(features), 1000, replace=False)
        features = features[idx]
        labels = labels[idx]
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.title('t-SNE Visualization of CNN Features')
    filename = "tsne_plot.png"
    plt.savefig(filename)
    plt.close()
    return filename

def plot_training_curves(history: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
    plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.legend()
    
    filename = "training_curves.png"
    plt.savefig(filename)
    plt.close()
    return filename

# --- Main Training & Experiment Loop ---

def train_and_evaluate(experiment_name: str, 
                       X_train, y_train, X_val, y_val, 
                       fold_idx: int):
    
    # Dataset & Loader
    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleCNN(n_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = []
    
    # Start MLflow Child Run (Fold Level)
    with mlflow.start_run(run_name=f"Fold_{fold_idx+1}", nested=True) as run:
        
        # Log Parameters
        mlflow.log_params({
            "fold": fold_idx + 1,
            "batch_size": BATCH_SIZE,
            "lr": 0.001,
            "optimizer": "Adam",
            "model": "SimpleCNN"
        })

        # Training Loop
        for epoch in range(EPOCHS):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation Step
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_labels = [], []
            all_features = [] # For t-SNE
            
            start_time = time.time() # Measure inference time
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs, features = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_features.extend(features.cpu().numpy())
            
            inference_time = (time.time() - start_time) / val_total
            
            # Calculate Metrics
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss, 'train_acc': epoch_train_acc,
                'val_loss': epoch_val_loss, 'val_acc': epoch_val_acc
            })
            
            # Log metrics per epoch
            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "train_acc": epoch_train_acc,
                "val_acc": epoch_val_acc
            }, step=epoch)
            
            logger.info(f"Fold {fold_idx+1} | Epoch {epoch+1}/{EPOCHS} | Val Acc: {epoch_val_acc:.4f}")

        # --- End of Training: Final Evaluation ---
        
        # Save History to CSV
        hist_df = pd.DataFrame(history)
        hist_df.to_csv("training_log.csv", index=False)
        mlflow.log_artifact("training_log.csv")
        
        # Generate & Log Plots
        curve_path = plot_training_curves(hist_df)
        mlflow.log_artifact(curve_path)
        
        cm_path = plot_confusion_matrix(all_labels, all_preds, CLASSES)
        mlflow.log_artifact(cm_path)
        
        tsne_path = plot_tsne(np.array(all_features), np.array(all_labels), CLASSES)
        mlflow.log_artifact(tsne_path)
        
        # Log Final Summary Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        mlflow.log_metrics({
            "final_accuracy": accuracy,
            "final_precision": precision,
            "final_recall": recall,
            "final_f1": f1,
            "inference_time_per_sample": inference_time
        })
        
        logger.info(f"Fold {fold_idx+1} Completed. Final F1: {f1:.4f}")

# --- Preparation Logic (Prevents Data Leakage) ---

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

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Setup Data
    df_meta = generate_mock_metadata(n_samples=150*4)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # 2. Define Experiments
    experiments = [
        "Baseline_LogMel",
        "Proposed_1_Denoised_LogMel",
        "Proposed_2_Raw_PCEN",
        # "Proposed_3_Mix" # Optional: Add logic for mixing if needed
    ]
    
    mlflow.set_experiment("Audio_Classification_Research")

    for exp_name in experiments:
        logger.info(f"Starting Experiment: {exp_name}")
        
        # Start MLflow Parent Run
        with mlflow.start_run(run_name=exp_name):
            mlflow.log_param("description", "4-Second Audio Classification")
            
            # Loop through Folds (Stratified by File ID)
            fold_iterator = skf.split(df_meta['file_id'], df_meta['label'])
            
            for fold, (train_idx, val_idx) in enumerate(fold_iterator):
                logger.info(f"--- Fold {fold+1}/{N_FOLDS} ---")
                
                # Prepare Data (Expensive step: Load -> Segment -> Feature)
                X_train, y_train, X_val, y_val = prepare_data_for_fold(
                    df_meta, train_idx, val_idx, exp_name
                )
                
                # Train & Log
                train_and_evaluate(exp_name, X_train, y_train, X_val, y_val, fold)
                
    logger.info("All Experiments Completed.")