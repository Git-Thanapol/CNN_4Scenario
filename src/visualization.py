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

def plot_training_curves(history: pd.DataFrame, save_dir=None):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
    plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.legend()
    
    filename = "training_curves.png"
    if save_dir:
        filename = os.path.join(save_dir, filename)
    plt.savefig(filename)
    plt.close()
    return filename
