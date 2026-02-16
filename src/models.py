import torch.nn as nn

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
