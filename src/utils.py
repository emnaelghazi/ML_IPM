"""
MAFAULDA Predictive Maintenance - Utility Functions
Helper functions used across the pipeline
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def parse_fault_from_path(file_path: Union[str, Path]) -> Tuple[str, Optional[str]]:
    """
    Extract fault type and severity from file path structure.
    
    Examples:
        data/raw/normal/12.288.csv -> ('normal', None)
        data/raw/horizontal-misalignment/0.5mm/13.1072.csv -> ('horizontal_misalignment', '0.5mm')
        data/raw/imbalance/6g/14.336.csv -> ('imbalance', '6g')
        data/raw/underhang/ball_fault/0g/15.1552.csv -> ('underhang_ball_fault', '0g')
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (fault_type, severity_level)
    """
    path = Path(file_path)
    parts = path.parts
    
    # Find the index where 'raw' appears
    try:
        raw_idx = parts.index('raw')
    except ValueError:
        raise ValueError(f"'raw' directory not found in path: {file_path}")
    
    # Get parts after 'raw'
    fault_parts = parts[raw_idx + 1:]
    
    if not fault_parts:
        raise ValueError(f"No fault information in path: {file_path}")
    
    # Handle different fault hierarchies
    if fault_parts[0] == 'normal':
        return 'normal', None
    
    elif fault_parts[0] in ['horizontal-misalignment', 'vertical-misalignment']:
        fault_type = fault_parts[0].replace('-', '_')
        severity = fault_parts[1] if len(fault_parts) > 1 else None
        return fault_type, severity
    
    elif fault_parts[0] == 'imbalance':
        severity = fault_parts[1] if len(fault_parts) > 1 else None
        return 'imbalance', severity
    
    elif fault_parts[0] in ['underhang', 'overhang']:
        # Handle bearing faults: underhang/ball_fault/0g/file.csv
        bearing_location = fault_parts[0]
        bearing_fault = fault_parts[1] if len(fault_parts) > 1 else 'unknown'
        severity = fault_parts[2] if len(fault_parts) > 2 else None
        
        fault_type = f"{bearing_location}_{bearing_fault}"
        return fault_type, severity
    
    else:
        # Unknown structure, use first part as fault type
        return fault_parts[0], None


def get_fault_label(file_path: Union[str, Path], fault_class_mapping: Dict[str, int]) -> int:
    """
    Get integer label for a fault from file path.
    
    Args:
        file_path: Path to CSV file
        fault_class_mapping: Dictionary mapping fault names to integer labels
        
    Returns:
        Integer label for the fault
    """
    fault_type, _ = parse_fault_from_path(file_path)
    
    if fault_type not in fault_class_mapping:
        raise ValueError(
            f"Fault type '{fault_type}' not found in mapping. "
            f"Available: {list(fault_class_mapping.keys())}"
        )
    
    return fault_class_mapping[fault_type]


def load_csv_safe(file_path: Union[str, Path], chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Safely load CSV file with error handling and optional chunking.
    
    Args:
        file_path: Path to CSV file
        chunk_size: If provided, load in chunks (for memory efficiency)
        
    Returns:
        DataFrame containing CSV data
    """
    try:
        if chunk_size:
            # Load in chunks and concatenate
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, header=None):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path, header=None)
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        file_path: Path to save pickle file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    
    logging.info(f"Saved pickle to {file_path}")


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    
    logging.info(f"Loaded pickle from {file_path}")
    return obj


def save_json(data: Dict, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
        indent: Indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    logging.info(f"Saved JSON to {file_path}")


def load_json(file_path: Union[str, Path]) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Loaded JSON from {file_path}")
    return data


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: Optional[np.ndarray] = None,
                     class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (for ROC-AUC)
        class_names: Names of classes for reporting
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    if class_names:
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                      output_dict=True, zero_division=0)
        metrics['per_class_metrics'] = report
    
    # ROC-AUC (if probabilities provided)
    if y_pred_proba is not None:
        try:
            # Multi-class ROC-AUC (one-vs-rest)
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Multi-class
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovo', average='macro')
        except Exception as e:
            logging.warning(f"Could not calculate ROC-AUC: {e}")
    
    # Business metrics
    cm = metrics['confusion_matrix']
    
    # False negatives (missed faults) - sum of off-diagonal elements in each row
    false_negatives = np.sum(cm, axis=1) - np.diag(cm)
    metrics['false_negatives_per_class'] = false_negatives
    metrics['total_false_negatives'] = np.sum(false_negatives)
    
    # False positives (false alarms)
    false_positives = np.sum(cm, axis=0) - np.diag(cm)
    metrics['false_positives_per_class'] = false_positives
    metrics['total_false_positives'] = np.sum(false_positives)
    
    # False negative rate and false positive rate
    total_samples = np.sum(cm, axis=1)
    metrics['false_negative_rate'] = false_negatives / (total_samples + 1e-10)
    metrics['false_positive_rate'] = false_positives / (total_samples + 1e-10)
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: Optional[Union[str, Path]] = None,
                         figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_learning_curves(train_scores: List[float], val_scores: List[float],
                        metric_name: str = 'Accuracy',
                        save_path: Optional[Union[str, Path]] = None,
                        figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot training and validation learning curves.
    
    Args:
        train_scores: Training scores over epochs/iterations
        val_scores: Validation scores over epochs/iterations
        metric_name: Name of the metric being plotted
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-o', label='Training', linewidth=2, markersize=4)
    plt.plot(epochs, val_scores, 'r-s', label='Validation', linewidth=2, markersize=4)
    
    # Calculate and display gap
    if len(train_scores) > 0 and len(val_scores) > 0:
        final_gap = abs(train_scores[-1] - val_scores[-1])
        plt.text(0.02, 0.98, f'Final Gap: {final_gap:.4f}',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Epoch / Iteration', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Learning Curves: {metric_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved learning curves to {save_path}")
    
    plt.close()


def check_overfitting(train_score: float, val_score: float, 
                     threshold: float = 0.05) -> bool:
    """
    Check if model is overfitting based on train-validation gap.
    
    Args:
        train_score: Training score
        val_score: Validation score
        threshold: Maximum acceptable gap
        
    Returns:
        True if overfitting detected
    """
    gap = abs(train_score - val_score)
    is_overfitting = gap > threshold
    
    if is_overfitting:
        logging.warning(f"Overfitting detected! Train: {train_score:.4f}, Val: {val_score:.4f}, Gap: {gap:.4f}")
    else:
        logging.info(f"No overfitting. Train: {train_score:.4f}, Val: {val_score:.4f}, Gap: {gap:.4f}")
    
    return is_overfitting


def get_timestamp() -> str:
    """Get current timestamp as string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 34m 12s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def create_synthetic_sample(n_samples: int = 25000, n_sensors: int = 8,
                           fault_type: str = 'normal',
                           noise_level: float = 0.1) -> pd.DataFrame:
    """
    Create synthetic sample data for testing.
    
    Args:
        n_samples: Number of samples (rows)
        n_sensors: Number of sensor columns
        fault_type: Type of fault to simulate
        noise_level: Level of noise to add
        
    Returns:
        DataFrame with synthetic sensor data
    """
    np.random.seed(42)
    
    # Generate base signals
    t = np.linspace(0, 1, n_samples)
    signals = []
    
    for i in range(n_sensors):
        if fault_type == 'normal':
            # Normal operation: low amplitude sinusoid + small noise
            signal = 0.5 * np.sin(2 * np.pi * 10 * t) + noise_level * np.random.randn(n_samples)
        
        elif fault_type == 'imbalance':
            # Imbalance: strong 1x RPM component
            signal = 2.0 * np.sin(2 * np.pi * 10 * t) + noise_level * np.random.randn(n_samples)
        
        elif fault_type == 'misalignment':
            # Misalignment: strong 2x RPM component
            signal = 1.5 * np.sin(2 * np.pi * 20 * t) + noise_level * np.random.randn(n_samples)
        
        elif 'bearing' in fault_type or 'ball_fault' in fault_type:
            # Bearing fault: impulsive signals
            signal = noise_level * np.random.randn(n_samples)
            # Add periodic impulses
            impulse_freq = 50  # impulses per second
            impulse_indices = np.arange(0, n_samples, n_samples // impulse_freq)
            signal[impulse_indices] += 5.0 * np.random.randn(len(impulse_indices))
        
        else:
            # Default: random signal
            signal = noise_level * np.random.randn(n_samples)
        
        signals.append(signal)
    
    # Create DataFrame
    df = pd.DataFrame(np.array(signals).T)
    
    return df


def memory_usage_mb() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in megabytes
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


if __name__ == "__main__":
    # Test utility functions
    logging.basicConfig(level=logging.INFO)
    
    # Test fault path parsing
    test_paths = [
        "data/raw/normal/12.288.csv",
        "data/raw/horizontal-misalignment/0.5mm/13.1072.csv",
        "data/raw/imbalance/6g/14.336.csv",
        "data/raw/underhang/ball_fault/0g/15.1552.csv",
    ]
    
    for path in test_paths:
        fault_type, severity = parse_fault_from_path(path)
        print(f"{path} -> Fault: {fault_type}, Severity: {severity}")
    
    # Test synthetic data generation
    print("\nGenerating synthetic samples...")
    for fault in ['normal', 'imbalance', 'misalignment', 'ball_fault']:
        df = create_synthetic_sample(fault_type=fault)
        print(f"{fault}: shape={df.shape}, mean={df.mean().mean():.4f}, std={df.std().mean():.4f}")
    
    # Test metrics calculation
    print("\nTesting metrics calculation...")
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    
    metrics = calculate_metrics(y_true, y_pred, class_names=['Class0', 'Class1', 'Class2'])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    
    print("\nUtility functions test complete!")