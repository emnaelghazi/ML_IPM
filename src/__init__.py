"""
MAFAULDA Predictive Maintenance System
Production-ready predictive maintenance using MAFAULDA dataset
"""

__version__ = "1.0.0"
__author__ = "MAFAULDA ML Team"

VERSION_INFO = {
    'version': __version__,
    'python_min': '3.10',
    'description': 'Industrial Predictive Maintenance with Anti-Overfitting Safeguards'
}

from .config import Config, get_config, reload_config
from .data_loader import MAFAULDADataLoader
from .utils import (
    parse_fault_from_path,
    get_fault_label,
    calculate_metrics,
    plot_confusion_matrix,
    plot_learning_curves
)

__all__ = [
    'Config',
    'get_config',
    'reload_config',
    'MAFAULDADataLoader',
    'parse_fault_from_path',
    'get_fault_label',
    'calculate_metrics',
    'plot_confusion_matrix',
    'plot_learning_curves',
    'VERSION_INFO'
]