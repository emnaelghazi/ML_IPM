"""
MAFAULDA Predictive Maintenance - Configuration Management
Loads and manages all configuration parameters from config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Path configuration for data and models."""
    raw_data: Path
    processed_data: Path
    splits: Path
    models: Path
    scalers: Path
    performance: Path
    logs: Path
    dashboard_uploads: Path
    
    def create_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for path_name, path_value in self.__dict__.items():
            path_value.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {path_value}")


@dataclass
class DataConfig:
    """Data handling configuration."""
    train_ratio: float
    val_ratio: float
    test_ratio: float
    stratify_by_fault_type: bool
    min_samples_per_class: int
    expected_rows_per_csv: int
    sensor_columns: int
    sensor_names: List[str]
    
    def __post_init__(self):
        """Validate configuration."""
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Train/val/test ratios must sum to 1.0"
        assert self.train_ratio > 0 and self.val_ratio > 0 and self.test_ratio > 0, \
            "All ratios must be positive"


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration."""
    n_jobs: int
    chunk_size: int
    batch_size: int
    time_domain: Dict[str, Any]
    frequency_domain: Dict[str, Any]
    time_frequency: Dict[str, Any]
    feature_selection: Dict[str, Any]
    scaling_method: str


@dataclass
class ModelConfig:
    """Individual model configuration."""
    enabled: bool
    hyperparameters: Dict[str, Any]
    cv_folds: Optional[int] = None
    n_iter: Optional[int] = None
    early_stopping_rounds: Optional[int] = None
    use_gpu: Optional[bool] = None
    architecture: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    cv_strategy: str
    cv_folds: int
    tuning_method: str
    scoring_metric: str
    overfitting_threshold: float
    max_training_time_hours: int
    checkpoint_frequency: int
    deterministic: bool


@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration."""
    metrics: List[str]
    plot_confusion_matrix: bool
    plot_roc_curves: bool
    plot_learning_curves: bool
    plot_feature_importance: bool
    cost_matrix: Dict[str, float]


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    theme: Dict[str, str]
    max_upload_size_mb: int
    prediction_confidence_threshold: float
    enable_shap_explanations: bool
    enable_pdf_export: bool
    auto_refresh: bool
    cache_ttl_seconds: int


@dataclass
class GPUConfig:
    """GPU configuration."""
    enabled: bool
    device: str
    memory_growth: bool
    mixed_precision: bool


class Config:
    """
    Main configuration class for MAFAULDA Predictive Maintenance System.
    Loads configuration from YAML file and provides structured access.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML configuration
        with open(self.config_path, 'r') as f:
            self._raw_config = yaml.safe_load(f)
        
        # Initialize structured configuration
        self._init_configuration()
        
        # Setup logging
        self._setup_logging()
        
        logging.info(f"Configuration loaded from {config_path}")
    
    def _init_configuration(self) -> None:
        """Initialize all configuration sections."""
        
        # Project info
        self.project_name = self._raw_config['project']['name']
        self.project_version = self._raw_config['project']['version']
        self.random_seed = self._raw_config['project']['random_seed']
        
        # Paths
        paths_dict = self._raw_config['paths']
        self.paths = PathConfig(
            raw_data=Path(paths_dict['raw_data']),
            processed_data=Path(paths_dict['processed_data']),
            splits=Path(paths_dict['splits']),
            models=Path(paths_dict['models']),
            scalers=Path(paths_dict['scalers']),
            performance=Path(paths_dict['performance']),
            logs=Path(paths_dict['logs']),
            dashboard_uploads=Path(paths_dict['dashboard_uploads'])
        )
        
        # Data configuration
        data_dict = self._raw_config['data']
        self.data = DataConfig(**data_dict)
        
        # Feature engineering
        self.feature_engineering = FeatureEngineeringConfig(
            **self._raw_config['feature_engineering']
        )
        
        # Models
        self.models = {}
        for model_name, model_config in self._raw_config['models'].items():
            if model_name == 'ensemble':
                # Ensemble has different structure - keep as dict
                self.models[model_name] = model_config
            else:
                # Regular models - convert to ModelConfig
                try:
                    self.models[model_name] = ModelConfig(**model_config)
                except TypeError as e:
                    # If ModelConfig fails, keep as dict (backward compatibility)
                    logging.warning(f"Could not create ModelConfig for {model_name}, using dict: {e}")
                    self.models[model_name] = model_config
        
        # Training
        self.training = TrainingConfig(**self._raw_config['training'])
        
        # Evaluation
        self.evaluation = EvaluationConfig(**self._raw_config['evaluation'])
        
        # Dashboard
        self.dashboard = DashboardConfig(**self._raw_config['dashboard'])
        
        # Logging
        self.logging_config = self._raw_config['logging']
        
        # GPU
        self.gpu = GPUConfig(**self._raw_config['gpu'])
        
        # Fault classes
        self.fault_classes = self._raw_config['fault_classes']
        self.num_classes = len(self.fault_classes)
        
        # Create reverse mapping (label -> name)
        self.fault_class_names = {v: k for k, v in self.fault_classes.items()}
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.logging_config
        
        # Create logs directory
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_format = log_config['format']
        log_level = getattr(logging, log_config['level'])
        
        handlers = []
        
        # File handler
        if log_config.get('file'):
            file_handler = logging.FileHandler(
                self.paths.logs / Path(log_config['file']).name
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        
        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(console_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers,
            force=True
        )
    
    def create_directories(self) -> None:
        """Create all required directories."""
        self.paths.create_directories()
        logging.info("All directories created successfully")
    
    def get_enabled_models(self) -> List[str]:
        """
        Get list of enabled models.
        
        Returns:
            List of enabled model names
        """
        enabled = []
        for model_name in self.models.keys():
            if self.is_model_enabled(model_name):
                enabled.append(model_name)
        return enabled
    
    def get_model_config(self, model_name: str):
        """
        Get configuration for specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig object or dict
            
        Raises:
            KeyError: If model not found
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found in configuration")
        return self.models[model_name]
    
    def is_model_enabled(self, model_name: str) -> bool:
        """
        Check if a model is enabled.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if enabled, False otherwise
        """
        if model_name not in self.models:
            return False
        
        model_config = self.models[model_name]
        
        if isinstance(model_config, ModelConfig):
            return model_config.enabled
        elif isinstance(model_config, dict):
            return model_config.get('enabled', False)
        
        return False
    
    def setup_reproducibility(self) -> None:
        """
        Setup reproducibility by setting random seeds.
        Must be called before any ML operations.
        """
        import random
        import numpy as np
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Try to set seeds for ML libraries
        try:
            import tensorflow as tf
            tf.random.set_seed(self.random_seed)
            logging.info(f"TensorFlow random seed set to {self.random_seed}")
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)
            logging.info(f"PyTorch random seed set to {self.random_seed}")
        except ImportError:
            pass
        
        # Set deterministic behavior if requested
        if self.training.deterministic:
            os.environ['PYTHONHASHSEED'] = str(self.random_seed)
            
            try:
                import tensorflow as tf
                tf.config.experimental.enable_op_determinism()
            except:
                pass
            
            try:
                import torch
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except:
                pass
        
        logging.info(f"Reproducibility setup complete (seed={self.random_seed})")
    
    def setup_gpu(self) -> None:
        """Configure GPU settings if available."""
        if not self.gpu.enabled:
            logging.info("GPU disabled in configuration")
            return
        
        # Try TensorFlow GPU setup
        try:
            import tensorflow as tf
            
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth
                    if self.gpu.memory_growth:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set visible devices
                    if self.gpu.device:
                        device_id = int(self.gpu.device.split(':')[-1])
                        tf.config.set_visible_devices(gpus[device_id], 'GPU')
                    
                    # Enable mixed precision
                    if self.gpu.mixed_precision:
                        from tensorflow.python.keras import mixed_precision
                        policy = mixed_precision.Policy('mixed_float16')
                        mixed_precision.set_global_policy(policy)
                    
                    logging.info(f"TensorFlow GPU configured: {len(gpus)} GPU(s) available")
                except RuntimeError as e:
                    logging.error(f"GPU setup error: {e}")
            else:
                logging.warning("No TensorFlow GPUs found")
        except ImportError:
            pass
        
        # Try PyTorch GPU setup
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                logging.info(f"PyTorch GPU available: {device_count} device(s), {device_name}")
                
                # Set default device
                if self.gpu.device:
                    torch.cuda.set_device(self.gpu.device)
            else:
                logging.warning("No PyTorch CUDA devices found")
        except ImportError:
            pass
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check if raw data directory exists
        if not self.paths.raw_data.exists():
            raise ValueError(f"Raw data directory does not exist: {self.paths.raw_data}")
        
        # Check if there are CSV files in raw data
        csv_files = list(self.paths.raw_data.rglob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.paths.raw_data}")
        
        logging.info(f"Found {len(csv_files)} CSV files in raw data directory")
        
        # Validate feature engineering settings
        if self.feature_engineering.n_jobs == -1:
            import multiprocessing
            actual_jobs = multiprocessing.cpu_count()
            logging.info(f"Using all {actual_jobs} CPU cores for parallel processing")
        
        # Validate model configurations
        enabled_models = self.get_enabled_models()
        if not enabled_models:
            raise ValueError("No models enabled in configuration")
        
        logging.info(f"Enabled models: {', '.join(enabled_models)}")
        
        # Check GPU availability if enabled
        if self.gpu.enabled:
            gpu_available = False
            
            try:
                import tensorflow as tf
                gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                try:
                    import torch
                    gpu_available = torch.cuda.is_available()
                except ImportError:
                    pass
            
            if not gpu_available:
                logging.warning("GPU enabled in config but no GPU detected. Will use CPU.")
        
        logging.info("Configuration validation passed")
        return True
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(project='{self.project_name}', "
            f"version='{self.project_version}', "
            f"models={self.get_enabled_models()}, "
            f"random_seed={self.random_seed})"
        )


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get or create configuration singleton instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def reload_config(config_path: str = "config.yaml") -> Config:
    """
    Force reload configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        New Config instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(config)
    print(f"\nEnabled models: {config.get_enabled_models()}")
    print(f"Number of classes: {config.num_classes}")
    print(f"Fault classes: {config.fault_classes}")
    
    # Validate configuration
    config.validate()
    
    # Create directories
    config.create_directories()
    
    # Setup reproducibility
    config.setup_reproducibility()
    
    # Setup GPU
    config.setup_gpu()