import logging
import mlflow
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from src.config import N_FOLDS, TRACKING_URI, EXPERIMENT_NAME, ARTIFACT_PATH
from src.data_prep import generate_mock_metadata, prepare_data_for_fold
from src.training import train_and_evaluate

from mlflow.tracking import MlflowClient

# Setup Logging
os.makedirs(ARTIFACT_PATH, exist_ok=True)
log_file_path = os.path.join(ARTIFACT_PATH, "experiment.log")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Setup Data
    df_meta = generate_mock_metadata(n_samples=150*4)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # 2. Define Experiments
    experiments = [
        "Baseline_LogMel",
        "Proposed_1_Denoised_Stationary_LogMel",
        "Proposed_2_Denoised_Nonstationary_LogMel",
        "Proposed_3_Raw_PCEN",
        # "Proposed_4_Mix" # Optional: Add logic for mixing if needed
    ]
    
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    client = MlflowClient()

    for exp_name in experiments:
        logger.info(f"Starting Experiment: {exp_name}")
        
        # Start MLflow Parent Run
        with mlflow.start_run(run_name=exp_name):
            mlflow.log_param("description", "4-Second Audio Classification")
            
            # Loop through Folds (Stratified by File ID)
            fold_iterator = skf.split(df_meta['file_id'], df_meta['label'])
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(fold_iterator):
                logger.info(f"--- Fold {fold+1}/{N_FOLDS} ---")
                
                # Prepare Data (Expensive step: Load -> Segment -> Feature)
                X_train, y_train, X_val, y_val = prepare_data_for_fold(
                    df_meta, train_idx, val_idx, exp_name
                )
                
                # Train & Log
                run_id, f1_score = train_and_evaluate(exp_name, X_train, y_train, X_val, y_val, fold)
                fold_results.append({'run_id': run_id, 'f1': f1_score, 'fold': fold+1})
                
            # Find and Tag Best Fold
            best_fold = max(fold_results, key=lambda x: x['f1'])
            logger.info(f"Best Fold for {exp_name}: Fold {best_fold['fold']} (F1: {best_fold['f1']:.4f})")
            
            client.set_tag(best_fold['run_id'], "best_fold", "True")
            
            # Log the log file as artifact
            mlflow.log_artifact(log_file_path)
                
    logger.info("All Experiments Completed.")