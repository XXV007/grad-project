"""
Evaluation Metrics Module
Provides comprehensive metrics for model evaluation

CPSC 589 - Multimodal Deepfake Detection
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_scores=None):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities (optional)
    
    Returns:
        metrics: Dictionary of metric values
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    metrics['tn'] = cm[0, 0]
    metrics['fp'] = cm[0, 1]
    metrics['fn'] = cm[1, 0]
    metrics['tp'] = cm[1, 1]
    
    # AUC-ROC if scores provided
    if y_scores is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['roc_thresholds'] = thresholds
    
    return metrics


def print_metrics(metrics):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'auc_roc' in metrics:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']:5d}  |  FP: {metrics['fp']:5d}")
    print(f"  FN: {metrics['fn']:5d}  |  TP: {metrics['tp']:5d}")
    print("="*50 + "\n")


def plot_confusion_matrix(cm, class_names=['Real', 'Fake'], save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def calculate_cross_dataset_metrics(results_dict):
    """
    Calculate metrics for cross-dataset evaluation
    
    Args:
        results_dict: Dictionary mapping dataset names to (y_true, y_pred, y_scores)
    
    Returns:
        summary: Dictionary of metrics per dataset
    """
    summary = {}
    
    for dataset_name, (y_true, y_pred, y_scores) in results_dict.items():
        metrics = calculate_metrics(y_true, y_pred, y_scores)
        summary[dataset_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics.get('auc_roc', None)
        }
    
    return summary


def print_cross_dataset_summary(summary):
    """
    Print cross-dataset evaluation summary
    
    Args:
        summary: Dictionary from calculate_cross_dataset_metrics
    """
    print("\n" + "="*70)
    print("CROSS-DATASET EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\n{'Dataset':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    for dataset, metrics in summary.items():
        print(f"{dataset:<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    # Test metrics calculation
    print("Testing Evaluation Metrics...")
    
    # Sample data
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.85, 0.95, 0.15])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    print_metrics(metrics)
    
    print("Metrics calculation test passed!")
