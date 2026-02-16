import torch
import numpy as np
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """PyTorch Dataset handling pre-computed features."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1) # Add channel dim
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
