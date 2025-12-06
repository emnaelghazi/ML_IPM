"""
MAFAULDA Predictive Maintenance - Model Prediction & Evaluation
Performs inference and comprehensive evaluation on test set
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Explanations will be limited.")

from config import Config
from utils import (
    load_pickle, save_pickle, save_json,
    calculate_metrics, plot_confusion_matrix,
    get_timestamp
)

warnings.filterwarnings('ignore')


class ModelPredictor:
    """
    Handles model inference and evaluation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize predictor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models = {}
        self.evaluation_results = {}
        
        logging.info("Initialized ModelPredictor")
    
    def load_models(self, models_dir: Path, timestamp: Optional[str] = None):
        """
        Load trained models.
        
        Args:
            models_dir: Directory containing models
            timestamp: Specific timestamp (uses latest if None)
        """
        if timestamp is None:
            # Find latest models
            model_files = sorted(models_dir.glob("*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No models found in {models_dir}")
            
            # Extract timestamp from first file
            timestamp = model_files[0].stem.rsplit('_', 1)[-1]
        
        logging.info(f"Loading models with timestamp: {timestamp}")
        
        # Load each model
        for model_file in models_dir.glob(f"*_{timestamp}.pkl"):
            if 'training_history' in model_file.name or 'best_params' in model_file.name:
                continue
            
            model_name = model_file.stem.rsplit('_', 1)[0]
            self.models[model_name] = load_pickle(model_file)
            logging.info(f"Loaded {model_name}")
        
        logging.info(f"Loaded {len(self.models)} models")
    
    def predict(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with a specific model.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model = self.models[model_name]
        
        # Handle XGBoost models specially
        model_type = type(model).__name__.lower()
        
        if 'xgboost' in model_type or 'booster' in str(type(model)):
            try:
                import xgboost as xgb
                
                # Convert to DMatrix for XGBoost
                dmatrix = xgb.DMatrix(X)
                
                # Get predictions
                y_pred_proba = model.predict(dmatrix)
                
                # For binary classification, XGBoost returns probabilities for class 1
                if y_pred_proba.ndim == 1:
                    # Binary classification
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    # Convert to 2D probabilities
                    y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
                else:
                    # Multi-class classification
                    y_pred = np.argmax(y_pred_proba, axis=1)
                
            except Exception as e:
                # Fallback to CPU prediction if DMatrix fails
                print(f"XGBoost GPU warning: {e}. Falling back to CPU prediction.")
                # Try direct prediction
                try:
                    y_pred_proba = model.predict(X)
                    if y_pred_proba.ndim == 1:
                        y_pred = (y_pred_proba > 0.5).astype(int)
                        y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
                    else:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                except:
                    # Last resort
                    y_pred = model.predict(X)
                    y_pred_proba = np.eye(self.config.num_classes)[y_pred]
        
        else:
            # For other models (Random Forest, etc.)
            y_pred = model.predict(X)
            
            # Probabilities
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
            else:
                # For models without predict_proba, use one-hot encoding
                y_pred_proba = np.eye(self.config.num_classes)[y_pred]
        
        return y_pred, y_pred_proba
    
    def evaluate_model(self, model_name: str, X: np.ndarray, y_true: np.ndarray,
                      split_name: str = 'test') -> Dict[str, Any]:
        """
        Evaluate a model comprehensively.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y_true: True labels
            split_name: Name of the split being evaluated
            
        Returns:
            Dictionary of evaluation results
        """
        logging.info(f"Evaluating {model_name} on {split_name} set...")
        
        # Make predictions
        y_pred, y_pred_proba = self.predict(model_name, X)
        
        # Calculate metrics
        class_names = [self.config.fault_class_names[i] for i in range(self.config.num_classes)]
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba, class_names)
        
        # Add model name and split
        metrics['model_name'] = model_name
        metrics['split'] = split_name
        
        # Log key metrics
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logging.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        # Store results
        if model_name not in self.evaluation_results:
            self.evaluation_results[model_name] = {}
        self.evaluation_results[model_name][split_name] = metrics
        
        # Plot confusion matrix
        self._plot_confusion_matrix(model_name, metrics, split_name)
        
        return metrics
    
    def evaluate_all_models(self, X: np.ndarray, y_true: np.ndarray,
                           split_name: str = 'test') -> Dict[str, Dict]:
        """
        Evaluate all loaded models.
        
        Args:
            X: Feature matrix
            y_true: True labels
            split_name: Name of the split
            
        Returns:
            Dictionary of all evaluation results
        """
        logging.info(f"\nEvaluating all models on {split_name} set...")
        logging.info("=" * 70)
        
        results = {}
        
        for model_name in self.models.keys():
            results[model_name] = self.evaluate_model(model_name, X, y_true, split_name)
        
        # Print comparison
        self._print_model_comparison(split_name)
        
        return results
    
    def _plot_confusion_matrix(self, model_name: str, metrics: Dict, split_name: str):
        """Plot and save confusion matrix."""
        cm = metrics['confusion_matrix']
        class_names = [self.config.fault_class_names[i] for i in range(self.config.num_classes)]
        
        save_path = self.config.paths.performance / f"confusion_matrix_{model_name}_{split_name}.png"
        plot_confusion_matrix(cm, class_names, save_path)
    
    def _print_model_comparison(self, split_name: str):
        """Print comparison table of all models."""
        logging.info(f"\nMODEL COMPARISON ({split_name.upper()} SET):")
        logging.info("-" * 90)
        logging.info(f"{'Model':<20} {'Accuracy':<12} {'F1-Macro':<12} {'F1-Weighted':<12} {'Precision':<12} {'Recall':<12}")
        logging.info("-" * 90)
        
        for model_name, results in self.evaluation_results.items():
            if split_name in results:
                metrics = results[split_name]
                logging.info(
                    f"{model_name:<20} "
                    f"{metrics['accuracy']:<12.4f} "
                    f"{metrics['f1_macro']:<12.4f} "
                    f"{metrics['f1_weighted']:<12.4f} "
                    f"{metrics['precision_macro']:<12.4f} "
                    f"{metrics['recall_macro']:<12.4f}"
                )
        
        logging.info("-" * 90)
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Complete report dictionary
        """
        report = {
            'timestamp': get_timestamp(),
            'configuration': {
                'num_classes': self.config.num_classes,
                'fault_classes': self.config.fault_classes,
                'random_seed': self.config.random_seed
            },
            'models_evaluated': list(self.models.keys()),
            'results': self.evaluation_results
        }
        
        # Find best model
        best_model = None
        best_f1 = 0
        
        for model_name, results in self.evaluation_results.items():
            if 'test' in results:
                f1 = results['test']['f1_macro']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
        
        report['best_model'] = {
            'name': best_model,
            'f1_macro': best_f1
        }
        
        # Business metrics
        if best_model and 'test' in self.evaluation_results[best_model]:
            test_metrics = self.evaluation_results[best_model]['test']
            report['business_metrics'] = {
                'false_negative_rate': float(np.mean(test_metrics['false_negative_rate'])),
                'false_positive_rate': float(np.mean(test_metrics['false_positive_rate'])),
                'total_false_negatives': int(test_metrics['total_false_negatives']),
                'total_false_positives': int(test_metrics['total_false_positives'])
            }
        
        return report
    
    def save_results(self, save_dir: Path):
        """
        Save evaluation results.
        
        Args:
            save_dir: Directory to save results
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = get_timestamp()
        
        # Save detailed results
        results_path = save_dir / f"evaluation_results_{timestamp}.pkl"
        save_pickle(self.evaluation_results, results_path)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        report_path = save_dir / f"evaluation_report_{timestamp}.json"
        save_json(report, report_path)
        
        logging.info(f"Saved evaluation results to {save_dir}")
        
        # Create summary visualization
        self._create_summary_plots(save_dir, timestamp)
    
    def _create_summary_plots(self, save_dir: Path, timestamp: str):
        """Create summary visualization plots."""
        
        # Model comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        metric_names = ['Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro']
        
        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            models = []
            train_scores = []
            test_scores = []
            
            for model_name, results in self.evaluation_results.items():
                if 'test' in results:
                    models.append(model_name)
                    test_scores.append(results['test'][metric])
                    # Add train score if available
                    if 'train' in results:
                        train_scores.append(results['train'][metric])
            
            x = np.arange(len(models))
            width = 0.35
            
            if train_scores:
                ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
                ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
            else:
                ax.bar(x, test_scores, width, label='Test', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"model_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Created summary plots")
    
    def explain_prediction(self, model_name: str, X: np.ndarray,
                          sample_idx: int = 0) -> Optional[Any]:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            sample_idx: Index of sample to explain
            
        Returns:
            SHAP values or None if not available
        """
        if not SHAP_AVAILABLE:
            logging.warning("SHAP not available for explanations")
            return None
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model = self.models[model_name]
        
        try:
            # Create explainer
            if hasattr(model, 'tree_'):  # Tree-based models
                explainer = shap.TreeExplainer(model)
            else:
                # Use kernel explainer for others (slower)
                explainer = shap.KernelExplainer(model.predict_proba, X[:100])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X[sample_idx:sample_idx+1])
            
            return shap_values
        
        except Exception as e:
            logging.error(f"SHAP explanation failed: {e}")
            return None


def main():
    """Main evaluation function."""
    from config import get_config
    
    # Setup
    config = get_config()
    config.setup_reproducibility()
    
    # Load test features
    logging.info("Loading test features...")
    test_df = pd.read_parquet(config.paths.processed_data / "features_test.parquet")
    
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values
    
    logging.info(f"Test set: {X_test.shape}")
    
    # Initialize predictor
    predictor = ModelPredictor(config)
    
    # Load models
    predictor.load_models(config.paths.models)
    
    # Evaluate all models on test set
    results = predictor.evaluate_all_models(X_test, y_test, split_name='test')
    
    # Save results
    predictor.save_results(config.paths.performance)
    
    # Generate and print report
    report = predictor.generate_comprehensive_report()
    
    logging.info(f"\n{'='*70}")
    logging.info("FINAL EVALUATION REPORT")
    logging.info(f"{'='*70}")
    logging.info(f"Best Model: {report['best_model']['name']}")
    logging.info(f"Best F1-Macro: {report['best_model']['f1_macro']:.4f}")
    
    if 'business_metrics' in report:
        bm = report['business_metrics']
        logging.info(f"\nBusiness Metrics:")
        logging.info(f"  False Negative Rate: {bm['false_negative_rate']:.2%}")
        logging.info(f"  False Positive Rate: {bm['false_positive_rate']:.2%}")
        logging.info(f"  Total False Negatives: {bm['total_false_negatives']}")
        logging.info(f"  Total False Positives: {bm['total_false_positives']}")
    
    logging.info(f"{'='*70}\n")
    logging.info("Evaluation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()