import logging
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .config import BATCH_SIZE, EPOCHS, CLASSES, DEVICE, ARTIFACT_PATH, PATIENCE
from .dataset import AudioDataset
from .models import SimpleCNN
from .visualization import plot_confusion_matrix, plot_tsne, plot_training_curves

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)

def train_and_evaluate(experiment_name: str, 
                       X_train, y_train, X_val, y_val, 
                       fold_idx: int):
    
    # Dataset & Loader
    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleCNN(n_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = []
    
    # Early Stopping Setup
    os.makedirs(ARTIFACT_PATH, exist_ok=True)
    checkpoint_path = os.path.join(ARTIFACT_PATH, f"model_{experiment_name}_fold{fold_idx}.pt")
    early_stopping = EarlyStopping(patience=PATIENCE, path=checkpoint_path)

    # Start MLflow Child Run (Fold Level)
    with mlflow.start_run(run_name=f"Fold_{fold_idx+1}", nested=True) as run:
        
        # Log Parameters
        mlflow.log_params({
            "fold": fold_idx + 1,
            "batch_size": BATCH_SIZE,
            "lr": 0.001,
            "optimizer": "Adam",
            "model": "SimpleCNN",
            "patience": PATIENCE
        })

        # Training Loop
        for epoch in range(EPOCHS):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation Step
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_labels = [], []
            all_features = [] # For t-SNE
            
            start_time = time.time() # Measure inference time
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs, features = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_features.extend(features.cpu().numpy())
            
            inference_time = (time.time() - start_time) / val_total
            
            # Calculate Metrics
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss, 'train_acc': epoch_train_acc,
                'val_loss': epoch_val_loss, 'val_acc': epoch_val_acc
            })
            
            # Log metrics per epoch
            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "train_acc": epoch_train_acc,
                "val_acc": epoch_val_acc
            }, step=epoch)
            
            logger.info(f"Fold {fold_idx+1} | Epoch {epoch+1}/{EPOCHS} | Val Acc: {epoch_val_acc:.4f}")

            # Call Early Stopping
            early_stopping(epoch_val_loss, model)
            
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(torch.load(checkpoint_path))
        
        # Log Model to MLflow
        mlflow.pytorch.log_model(model, "model")
        
        # Get Run ID
        run_id = run.info.run_id

        # --- End of Training: Final Evaluation ---
        
        # Save History to CSV
        os.makedirs(ARTIFACT_PATH, exist_ok=True)
        csv_path = os.path.join(ARTIFACT_PATH, "training_log.csv")
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        # Generate & Log Plots
        curve_path = plot_training_curves(hist_df, save_dir=ARTIFACT_PATH)
        mlflow.log_artifact(curve_path)
        
        cm_path = plot_confusion_matrix(all_labels, all_preds, CLASSES, save_dir=ARTIFACT_PATH)
        mlflow.log_artifact(cm_path)
        
        tsne_path = plot_tsne(np.array(all_features), np.array(all_labels), CLASSES, save_dir=ARTIFACT_PATH)
        mlflow.log_artifact(tsne_path)
        
        # Log Final Summary Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        mlflow.log_metrics({
            "final_accuracy": accuracy,
            "final_precision": precision,
            "final_recall": recall,
            "final_f1": f1,
            "inference_time_per_sample": inference_time
        })
        
        logger.info(f"Fold {fold_idx+1} Completed. Final F1: {f1:.4f}")
        
        return run_id, f1
