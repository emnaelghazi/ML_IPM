"""
MAFAULDA Predictive Maintenance - Model Training
Trains multiple models with hyperparameter tuning and cross-validation
"""

import logging
from sys import version
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb
import lightgbm as lgb

from config import Config
from utils import (
    save_pickle, load_pickle, calculate_metrics, 
    plot_confusion_matrix, plot_learning_curves,
    check_overfitting, format_duration, get_timestamp
)

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Trains and evaluates machine learning models with rigorous overfitting prevention.
    """
    
    def __init__(self, config: Config):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models = {}
        self.best_params = {}
        self.training_history = {}
        
        logging.info("Initialized ModelTrainer")
    
    def _get_model_config_value(self, model_name: str, key: str, default=None):
        """
        Safely get configuration value from model config (handles both dict and ModelConfig).
        
        Args:
            model_name: Name of the model
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        model_config = self.config.models.get(model_name)
        if model_config is None:
            return default
        
        if hasattr(model_config, key):
            return getattr(model_config, key)
        elif isinstance(model_config, dict):
            return model_config.get(key, default)
        
        return default
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> RandomForestClassifier:
        """
        Train Random Forest with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model
        """
        logging.info("Training Random Forest...")
        start_time = time.time()
        
        # Get config values safely
        hyperparameters = self._get_model_config_value('random_forest', 'hyperparameters', {})
        cv_folds = self._get_model_config_value('random_forest', 'cv_folds', 5)
        n_iter = self._get_model_config_value('random_forest', 'n_iter', 50)
        
        # Base model
        rf = RandomForestClassifier(
            random_state=self.config.random_seed,
            n_jobs=-1,
            verbose=0
        )
        
        # Hyperparameter search
        param_distributions = hyperparameters
        
        # Stratified CV
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.config.random_seed
        )
        
        # Scorer
        scorer = make_scorer(f1_score, average='macro')
        
        # Random search
        search = RandomizedSearchCV(
            rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            random_state=self.config.random_seed,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        # Best model
        best_rf = search.best_estimator_
        self.best_params['random_forest'] = search.best_params_
        
        # Evaluate on validation
        train_score = best_rf.score(X_train, y_train)
        val_score = best_rf.score(X_val, y_val)
        
        # Check overfitting
        check_overfitting(train_score, val_score, self.config.training.overfitting_threshold)
        
        # Store training history
        self.training_history['random_forest'] = {
            'train_score': train_score,
            'val_score': val_score,
            'best_params': search.best_params_,
            'training_time': time.time() - start_time
        }
        
        logging.info(f"Random Forest trained in {format_duration(time.time() - start_time)}")
        logging.info(f"Train accuracy: {train_score:.4f}, Val accuracy: {val_score:.4f}")
        
        return best_rf
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
        """
        Train XGBoost with GPU acceleration and early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model
        """
        logging.info("Training XGBoost...")
        start_time = time.time()
        
        # Get config values safely
        use_gpu = self._get_model_config_value('xgboost', 'use_gpu', False)
        hyperparameters = self._get_model_config_value('xgboost', 'hyperparameters', {})
        cv_folds = self._get_model_config_value('xgboost', 'cv_folds', 5)
        n_iter = self._get_model_config_value('xgboost', 'n_iter', 50)
        early_stopping_rounds = self._get_model_config_value('xgboost', 'early_stopping_rounds', 50)
        
        # GPU settings - UPDATED FOR XGBOOST 3.1+
        xgb_version = xgb.__version__
        logging.info(f"XGBoost version: {xgb_version}")
        
        # Simple version check (no packaging module needed)
        # Check if version is 3.1.0 or higher by parsing the string
        version_parts = xgb_version.split('.')
        major_version = int(version_parts[0])
        minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        is_new_api = (major_version > 3) or (major_version == 3 and minor_version >= 1)
        
        if use_gpu and self.config.gpu.enabled:
            tree_method = 'hist'
            if is_new_api:
                # New parameter for XGBoost 3.1+
                device = 'cuda'
                # Remove gpu_id from hyperparameters if present
                if 'gpu_id' in hyperparameters:
                    del hyperparameters['gpu_id']
            else:
                # Old parameter for XGBoost < 3.1
                tree_method = 'gpu_hist'
                gpu_id = 0
        else:
            tree_method = 'hist'
            if is_new_api:
                device = 'cpu'
            else:
                gpu_id = -1
        
        # Base model - UPDATED FOR XGBOOST 3.1+
        model_params = {
            'objective': 'multi:softmax',
            'num_class': self.config.num_classes,
            'tree_method': tree_method,
            'random_state': self.config.random_seed,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': early_stopping_rounds
        }
        
        # Add device parameter based on version
        if is_new_api:
            model_params['device'] = device
        else:
            model_params['gpu_id'] = gpu_id
        
        xgb_model = xgb.XGBClassifier(**model_params)
        
        # Hyperparameter search
        param_distributions = hyperparameters.copy()
        
        # Remove non-standard parameters
        if 'scale_pos_weight' in param_distributions:
            del param_distributions['scale_pos_weight']
        
        # Remove gpu_id if it's still there (for newer versions)
        if 'gpu_id' in param_distributions and is_new_api:
            del param_distributions['gpu_id']
        
        # Stratified CV
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.config.random_seed
        )
        
        # Scorer
        scorer = make_scorer(f1_score, average='macro')
        
        # Fit parameters for eval_set
        fit_params = {
            'eval_set': [(X_val, y_val)],
            'verbose': False
        }
        
        # Random search
        search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            random_state=self.config.random_seed,
            n_jobs=1,  # XGBoost handles parallelism internally
            verbose=1,
            error_score='raise'  # This will help debug any other issues
        )
        
        try:
            search.fit(
                X_train, y_train,
                **fit_params
            )
        except Exception as e:
            logging.error(f"XGBoost training failed: {str(e)}")
            # Fallback to CPU if GPU fails
            logging.info("Attempting fallback to CPU...")
            if is_new_api:
                model_params['device'] = 'cpu'
            else:
                model_params['gpu_id'] = -1
                model_params['tree_method'] = 'hist'
            
            xgb_model = xgb.XGBClassifier(**model_params)
            search = RandomizedSearchCV(
                xgb_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scorer,
                cv=cv,
                random_state=self.config.random_seed,
                n_jobs=1,
                verbose=1,
                error_score='raise'
            )
            search.fit(X_train, y_train, **fit_params)
        
        # Best model
        best_xgb = search.best_estimator_
        self.best_params['xgboost'] = search.best_params_
        
        # Evaluate
        train_score = best_xgb.score(X_train, y_train)
        val_score = best_xgb.score(X_val, y_val)
        
        check_overfitting(train_score, val_score, self.config.training.overfitting_threshold)
        
        # Store history
        self.training_history['xgboost'] = {
            'train_score': train_score,
            'val_score': val_score,
            'best_params': search.best_params_,
            'training_time': time.time() - start_time,
            'best_iteration': best_xgb.best_iteration if hasattr(best_xgb, 'best_iteration') else None,
            'xgb_version': xgb_version,
            'device_used': device if is_new_api else f"gpu_id={gpu_id}"
        }
        
        logging.info(f"XGBoost trained in {format_duration(time.time() - start_time)}")
        logging.info(f"Train accuracy: {train_score:.4f}, Val accuracy: {val_score:.4f}")
        
        return best_xgb
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> lgb.LGBMClassifier:
        """
        Train LightGBM with GPU acceleration and early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model
        """
        logging.info("Training LightGBM...")
        start_time = time.time()
        
        # Get config values safely
        use_gpu = self._get_model_config_value('lightgbm', 'use_gpu', False)
        hyperparameters = self._get_model_config_value('lightgbm', 'hyperparameters', {})
        cv_folds = self._get_model_config_value('lightgbm', 'cv_folds', 5)
        n_iter = self._get_model_config_value('lightgbm', 'n_iter', 50)
        early_stopping_rounds = self._get_model_config_value('lightgbm', 'early_stopping_rounds', 50)
        
        # GPU settings
        if use_gpu and self.config.gpu.enabled:
            device = 'gpu'
            gpu_platform_id = 0
            gpu_device_id = 0
        else:
            device = 'cpu'
            gpu_platform_id = -1
            gpu_device_id = -1
        
        # Base model
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=self.config.num_classes,
            device=device,
            gpu_platform_id=gpu_platform_id,
            gpu_device_id=gpu_device_id,
            random_state=self.config.random_seed,
            verbose=-1
        )
        
        # Hyperparameter search
        param_distributions = hyperparameters.copy()
        
        # Remove non-standard parameters
        if 'class_weight' in param_distributions:
            del param_distributions['class_weight']
        
        # Stratified CV
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.config.random_seed
        )
        
        # Scorer
        scorer = make_scorer(f1_score, average='macro')
        
        # Random search
        search = RandomizedSearchCV(
            lgb_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            random_state=self.config.random_seed,
            n_jobs=1,
            verbose=1
        )
        
        search.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        
        # Best model
        best_lgb = search.best_estimator_
        self.best_params['lightgbm'] = search.best_params_
        
        # Evaluate
        train_score = best_lgb.score(X_train, y_train)
        val_score = best_lgb.score(X_val, y_val)
        
        check_overfitting(train_score, val_score, self.config.training.overfitting_threshold)
        
        # Store history
        self.training_history['lightgbm'] = {
            'train_score': train_score,
            'val_score': val_score,
            'best_params': search.best_params_,
            'training_time': time.time() - start_time,
            'best_iteration': best_lgb.best_iteration_ if hasattr(best_lgb, 'best_iteration_') else None
        }
        
        logging.info(f"LightGBM trained in {format_duration(time.time() - start_time)}")
        logging.info(f"Train accuracy: {train_score:.4f}, Val accuracy: {val_score:.4f}")
        
        return best_lgb
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> VotingClassifier:
        """
        Train ensemble of top models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Ensemble model
        """
        logging.info("Training Ensemble...")
        start_time = time.time()
        
        ensemble_config = self.config.models['ensemble']
        base_models = ensemble_config['base_models']
        
        # Collect trained models
        estimators = []
        for model_name in base_models:
            if model_name in self.models:
                # SPECIAL FIX FOR XGBOOST: Clone and remove early_stopping_rounds
                if model_name == 'xgboost':
                    from copy import deepcopy
                    xgb_model = deepcopy(self.models[model_name])
                    
                    # Remove early_stopping_rounds for ensemble training
                    # This prevents the "Must have at least 1 validation dataset" error
                    xgb_model.set_params(
                        early_stopping_rounds=None,
                        eval_set=None
                    )
                    
                    # Also clear any stored evaluation data
                    if hasattr(xgb_model, '_Booster') and xgb_model._Booster is not None:
                        # Try to reset the booster
                        try:
                            xgb_model._Booster = None
                        except:
                            pass
                    
                    estimators.append((model_name, xgb_model))
                else:
                    estimators.append((model_name, self.models[model_name]))
            else:
                logging.warning(f"Model {model_name} not trained, skipping from ensemble")
        
        if len(estimators) < 2:
            logging.error("Need at least 2 models for ensemble")
            return None
        
        # Create ensemble
        if ensemble_config['weights'] == 'auto':
            # Weight by validation performance
            weights = []
            for model_name, model in estimators:
                if model_name in self.training_history:
                    val_score = self.training_history[model_name]['val_score']
                    weights.append(val_score)
                else:
                    weights.append(1.0)  # Default weight if no history
            weights = np.array(weights) / sum(weights)
        else:
            weights = None
        
        # Train each model individually first (to avoid parallel fitting issues with XGBoost)
        logging.info("Pre-fitting models for ensemble...")
        for model_name, model in estimators:
            if model_name == 'xgboost':
                # For XGBoost, fit without early stopping
                model.fit(X_train, y_train)
                logging.info(f"Pre-fit {model_name}")
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=ensemble_config['voting'],
            weights=weights,
            n_jobs=-1
        )
        
        # Train ensemble
        try:
            ensemble.fit(X_train, y_train)
        except ValueError as e:
            if "Must have at least 1 validation dataset for early stopping" in str(e):
                logging.warning("XGBoost early stopping conflict detected. Retrying with XGBoost excluded...")
                # Alternative: Exclude XGBoost and retry
                estimators_without_xgb = [(name, model) for name, model in estimators if name != 'xgboost']
                if len(estimators_without_xgb) >= 2:
                    ensemble = VotingClassifier(
                        estimators=estimators_without_xgb,
                        voting=ensemble_config['voting'],
                        weights=weights[:len(estimators_without_xgb)] if weights is not None else None,
                        n_jobs=-1
                    )
                    ensemble.fit(X_train, y_train)
                    # Update base_models to reflect which models were actually used
                    base_models = [name for name, _ in estimators_without_xgb]
                else:
                    logging.error("Not enough models after excluding XGBoost")
                    return None
            else:
                raise e
        
        # Evaluate
        train_score = ensemble.score(X_train, y_train)
        val_score = ensemble.score(X_val, y_val)
        
        check_overfitting(train_score, val_score, self.config.training.overfitting_threshold)
        
        # Store history
        self.training_history['ensemble'] = {
            'train_score': train_score,
            'val_score': val_score,
            'base_models': base_models,  # Use the actual models used
            'weights': weights.tolist() if weights is not None else None,
            'training_time': time.time() - start_time
        }
        
        logging.info(f"Ensemble trained in {format_duration(time.time() - start_time)}")
        logging.info(f"Train accuracy: {train_score:.4f}, Val accuracy: {val_score:.4f}")
        logging.info(f"Models in ensemble: {base_models}")
        
        return ensemble
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train all enabled models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of trained models
        """
        logging.info("\n" + "="*70)
        logging.info("TRAINING ALL MODELS")
        logging.info("="*70)
        
        total_start = time.time()
        
        # Train Random Forest
        if self.config.is_model_enabled('random_forest'):
            self.models['random_forest'] = self.train_random_forest(X_train, y_train, X_val, y_val)
        
        # Train XGBoost
        if self.config.is_model_enabled('xgboost'):
            self.models['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Train LightGBM
        if self.config.is_model_enabled('lightgbm'):
            self.models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Train Ensemble (if enabled and base models available)
        if self.config.is_model_enabled('ensemble') and len(self.models) >= 2:
            self.models['ensemble'] = self.train_ensemble(X_train, y_train, X_val, y_val)
        
        total_time = time.time() - total_start
        
        logging.info("\n" + "="*70)
        logging.info(f"ALL MODELS TRAINED IN {format_duration(total_time)}")
        logging.info("="*70)
        
        # Print summary
        self._print_training_summary()
        
        return self.models
    
    def _print_training_summary(self) -> None:
        """Print summary of all trained models."""
        logging.info("\nTRAINING SUMMARY:")
        logging.info("-" * 70)
        logging.info(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10} {'Time':<15}")
        logging.info("-" * 70)
        
        for model_name, history in self.training_history.items():
            train_acc = history['train_score']
            val_acc = history['val_score']
            gap = abs(train_acc - val_acc)
            training_time = format_duration(history['training_time'])
            
            logging.info(
                f"{model_name:<20} {train_acc:<12.4f} {val_acc:<12.4f} "
                f"{gap:<10.4f} {training_time:<15}"
            )
        
        logging.info("-" * 70)
    
    def save_models(self, save_dir: Path) -> None:
        """
        Save all trained models.
        
        Args:
            save_dir: Directory to save models
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = get_timestamp()
        
        for model_name, model in self.models.items():
            model_path = save_dir / f"{model_name}_{timestamp}.pkl"
            save_pickle(model, model_path)
        
        # Save training history
        history_path = save_dir / f"training_history_{timestamp}.pkl"
        save_pickle(self.training_history, history_path)
        
        # Save best params
        params_path = save_dir / f"best_params_{timestamp}.pkl"
        save_pickle(self.best_params, params_path)
        
        logging.info(f"Saved all models to {save_dir}")
    
    def load_models(self, load_dir: Path, timestamp: Optional[str] = None) -> None:
        """
        Load trained models.
        
        Args:
            load_dir: Directory containing models
            timestamp: Specific timestamp to load (uses latest if None)
        """
        if timestamp is None:
            # Find latest timestamp
            history_files = sorted(load_dir.glob("training_history_*.pkl"))
            if not history_files:
                raise FileNotFoundError(f"No training history found in {load_dir}")
            timestamp = history_files[-1].stem.split('_')[-1]
        
        logging.info(f"Loading models with timestamp: {timestamp}")
        
        # Load models
        for model_file in load_dir.glob(f"*_{timestamp}.pkl"):
            if 'training_history' in model_file.name or 'best_params' in model_file.name:
                continue
            
            model_name = model_file.stem.rsplit('_', 1)[0]
            self.models[model_name] = load_pickle(model_file)
            logging.info(f"Loaded {model_name}")
        
        # Load history
        history_path = load_dir / f"training_history_{timestamp}.pkl"
        if history_path.exists():
            self.training_history = load_pickle(history_path)
        
        # Load best params
        params_path = load_dir / f"best_params_{timestamp}.pkl"
        if params_path.exists():
            self.best_params = load_pickle(params_path)
        
        logging.info(f"Loaded {len(self.models)} models")


def main():
    """Main training function."""
    from config import get_config
    
    # Setup
    config = get_config()
    config.setup_reproducibility()
    config.setup_gpu()
    
    # Load preprocessed features
    logging.info("Loading preprocessed features...")
    
    train_df = pd.read_parquet(config.paths.processed_data / "features_train.parquet")
    val_df = pd.read_parquet(config.paths.processed_data / "features_val.parquet")
    
    # Separate features and labels
    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    X_val = val_df.drop(columns=['label']).values
    y_val = val_df['label'].values
    
    logging.info(f"Training set: {X_train.shape}")
    logging.info(f"Validation set: {X_val.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train all models
    models = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Save models
    trainer.save_models(config.paths.models)
    
    logging.info("\nTraining pipeline complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()