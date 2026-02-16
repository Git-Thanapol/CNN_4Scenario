import logging
import mlflow
from sklearn.model_selection import StratifiedKFold
from src.config import N_FOLDS
from src.data_prep import generate_mock_metadata, prepare_data_for_fold
from src.training import train_and_evaluate

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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