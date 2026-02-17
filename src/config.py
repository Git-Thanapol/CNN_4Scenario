import torch

SAMPLE_RATE = 22050
DURATION = 4.0  # seconds
WINDOW_SIZE = 1.0  # seconds
STRIDE = 0.5  # seconds
N_MELS = 128
HOP_LENGTH = 512
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.0001
N_FOLDS = 5
CLASSES = ['healthy', 'imbalance', 'fouling', 'seaweed']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRACKING_URI = "http://localhost:5000" # Prod
EXPERIMENT_NAME = "Lower_Sr_Bandpass"
ARTIFACT_PATH = "mlartifacts"
DATAFOLDER = "~/notebooks/propeller_audio_records/raw_train_data"
LOW_PASS_CUTOFF = 11000 # Hz
HIGH_PASS_CUTOFF = 2000 # Hz

# Augmentation Flags
AUGMENT_RAW_DATA = True # Apply TimeShift, PitchShift, Noise etc. during data prep
AUGMENT_SPECTROGRAM = True # Apply Masking during training (on-the-fly)
# (Added flags for augmentation)