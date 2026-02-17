import torch
import numpy as np
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """PyTorch Dataset handling pre-computed features."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1) # Add channel dim
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx]
