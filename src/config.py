import torch

SAMPLE_RATE = 22050
DURATION = 4.0  # seconds
WINDOW_SIZE = 1.0  # seconds
STRIDE = 0.5  # seconds
N_MELS = 128
HOP_LENGTH = 512
BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 10
LEARNING_RATE = 0.0001
N_FOLDS = 5
CLASSES = ['healthy', 'imbalance', 'fouling', 'seaweed']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRACKING_URI = "http://localhost:5000" # Prod
EXPERIMENT_NAME = "Model_Comparison_Study_Stationary_2026_June_17"
ARTIFACT_PATH = "mlartifacts"
#DATAFOLDER = "~/notebooks/propeller_audio_records/raw_train_data"
DATAFOLDER = "~/notebooks/202606_Experiments/Dataset/TrainingData_202606"#
LOW_PASS_CUTOFF = 11000 # Hz
NOISE_PROFILE_PATH = "~/notebooks/202606_Experiments/Dataset/NoiseProfile/Audio_Healthy_1500_20251211_113603.wav"  # Reference noise profile for stationary denoising
NOISE_PROFILE_FOR_FOULING = "~/notebooks/202606_Experiments/Dataset/NoiseProfile/TRIM_TANK_SOUND_PWM1500_Iter7.wav"  # Reference noise profile for stationary denoising
HIGH_PASS_CUTOFF = 2000 # Hz
DROPOUT_RATE = 0.5

# Augmentation Flags
AUGMENT_RAW_DATA = True # Apply TimeShift, PitchShift, Noise etc. during data prep
AUGMENT_SPECTROGRAM = True # Apply Masking during training (on-the-fly)
# (Added flags for augmentation)