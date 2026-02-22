import numpy as np
import logging
import mlflow
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import src.config as config
from src.config import N_FOLDS, TRACKING_URI, EXPERIMENT_NAME, ARTIFACT_PATH, DATAFOLDER
from src.data_prep import prepare_data_for_fold, load_metadata
from src.training import train_and_evaluate

from mlflow.tracking import MlflowClient

# Setup Logging
os.makedirs(ARTIFACT_PATH, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(ARTIFACT_PATH, f"experiment_comparison_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MLflow Setup
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# Experiment Definitions
EXPERIMENTS_LIST = [

    #CNN Simple
    {"name": "CNN_Simple_Stationary", "model": "SimpleCNN", "process": "Stationary"},
    # CNN + MLP
    #{"name": "CNN_MLP_PCEN", "model": "CNN_MLP", "process": "PCEN"},
    {"name": "CNN_MLP_Stationary", "model": "CNN_MLP", "process": "Stationary"},
    #{"name": "CNN_MLP_NonStationary", "model": "CNN_MLP", "process": "NonStationary"},

    # CNN + Attention
    #{"name": "CNN_Attention_PCEN", "model": "CNN_Attention", "process": "PCEN"},
    {"name": "CNN_Attention_Stationary", "model": "CNN_Attention", "process": "Stationary"},
    #{"name": "CNN_Attention_NonStationary", "model": "CNN_Attention", "process": "NonStationary"},
    
    # VGG
    #{"name": "VGG_PCEN", "model": "VGG", "process": "PCEN"},
    {"name": "VGG_Stationary", "model": "VGG", "process": "Stationary"},
    #{"name": "VGG_NonStationary", "model": "VGG", "process": "NonStationary"},

    # AST
    #{"name": "AST_AFSC_ResCNN", "model": "AST_AFSC_ResCNN", "process": "Bandpass_LogMel", 
    #"extra_args": {"ast_weights": r"C:\Users\Thana\dev\cnn_comparison\AST\audioset_10_10_0.4593 (1).pth"}}
]

# Override the global experiment name for this suite
COMPARISON_EXPERIMENT_NAME = "Stationary_Tank_Test_NoiseProfile"

def run_comparison():
    try:
        # Check if experiment exists, create if not
        try:
            experiment_id = mlflow.create_experiment(COMPARISON_EXPERIMENT_NAME)
        except Exception:
            experiment_id = mlflow.get_experiment_by_name(COMPARISON_EXPERIMENT_NAME).experiment_id
        
        mlflow.set_experiment(COMPARISON_EXPERIMENT_NAME)
        
        # Load Metadata ONE TIME
        df_meta = load_metadata(DATAFOLDER)
        
        # Define Folds based on File IDs (Leakage Prevention)
        unique_files = df_meta['file_id'].unique()
        unique_labels = df_meta.drop_duplicates('file_id')['label']
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        
        fold_splits = list(skf.split(unique_files, unique_labels))
        
        for exp_config in EXPERIMENTS_LIST:
            exp_name = exp_config["name"]
            model_arch = exp_config["model"]
            process_type = exp_config["process"]
            extra_args = exp_config.get("extra_args", None)
            
            logger.info(f"=== Starting Experiment: {exp_name} (Model: {model_arch}) ===")
            
            # Start Parent Run for this specific Experiment Configuration
            with mlflow.start_run(run_name=exp_name, nested=False) as parent_run:
                mlflow.log_param("experiment_group", "comparison")
                mlflow.log_param("preprocessing", process_type)
                
                f1_scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                    logger.info(f"  Running Fold {fold_idx+1}/{N_FOLDS}...")
                    
                    # Prepare Data
                    # Note: process_type string determines the logic in data_prep
                    # augment=True for training sets
                    X_train, y_train, X_val, y_val = prepare_data_for_fold(
                        df_meta, train_idx, val_idx, 
                        experiment_type=process_type, 
                        augment=True
                    )
                    
                    # Train & Evaluate
                    run_id, val_f1 = train_and_evaluate(
                        experiment_name=exp_name, # Passed for checkpoint naming
                        X_train=X_train, y_train=y_train, 
                        X_val=X_val, y_val=y_val, 
                        fold_idx=fold_idx,
                        model_arch=model_arch,
                        extra_args=extra_args
                    )
                    f1_scores.append(val_f1)
                
                avg_f1 = np.mean(f1_scores)
                mlflow.log_metric("avg_f1", avg_f1)
                logger.info(f"=== Experiment {exp_name} Finished. Avg F1: {avg_f1:.4f} ===\n")

    except Exception as e:
        logger.error(f"Comparison Experiment Failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    run_comparison()
