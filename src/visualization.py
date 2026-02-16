import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def plot_confusion_matrix(y_true, y_pred, classes, save_dir=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    filename = "confusion_matrix.png"
    if save_dir:
        filename = os.path.join(save_dir, filename)
    plt.savefig(filename)
    plt.close()
    return filename

def plot_tsne(features, labels, classes, save_dir=None):
    """Plots t-SNE of the extracted features."""
    if len(features) > 1000: # Subsample for speed if needed
        idx = np.random.choice(len(features), 1000, replace=False)
        features = features[idx]
        labels = labels[idx]
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.title('t-SNE Visualization of CNN Features')
    filename = "tsne_plot.png"
    if save_dir:
        filename = os.path.join(save_dir, filename)
    plt.savefig(filename)
    plt.close()
    return filename

def plot_metrics_separately(history: pd.DataFrame, save_dir=None):
    """
    Plots Loss, Accuracy, Precision, Recall, and F1 separately.
    Returns a list of generated file paths.
    """
    metrics = [
        ('train_loss', 'val_loss', 'Loss per Epoch', 'loss_curve.png'),
        ('train_acc', 'val_acc', 'Accuracy per Epoch', 'accuracy_curve.png'),
        (None, 'val_precision', 'Validation Precision per Epoch', 'precision_curve.png'),
        (None, 'val_recall', 'Validation Recall per Epoch', 'recall_curve.png'),
        (None, 'val_f1', 'Validation F1 Score per Epoch', 'f1_curve.png')
    ]
    
    generated_files = []
    
    for train_metric, val_metric, title, filename in metrics:
        plt.figure(figsize=(10, 6))
        
        if train_metric and train_metric in history.columns:
            plt.plot(history['epoch'], history[train_metric], label=f'Train {train_metric.split("_")[1].capitalize()}')
            
        if val_metric and val_metric in history.columns:
            plt.plot(history['epoch'], history[val_metric], label=f'Val {val_metric.split("_")[1].capitalize()}')
            
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = filename
            
        plt.savefig(filepath)
        plt.close()
        generated_files.append(filepath)
        
    return generated_files
