import torch

SAMPLE_RATE = 44100
DURATION = 4.0  # seconds
WINDOW_SIZE = 1.0  # seconds
STRIDE = 0.5  # seconds
N_MELS = 128
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 20
N_FOLDS = 5
CLASSES = ['healthy', 'imbalance', 'fouling', 'seaweed']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRACKING_URI = "http://localhost:5000" # Prod
EXPERIMENT_NAME = "Audio_Classification_Research_Test"
ARTIFACT_PATH = "mlartifacts"
DATAFOLDER = "~/notebooks/propeller_audio_records/raw_train_data"
LOW_PASS_CUTOFF = 900 # Hz
