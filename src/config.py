import torch

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
