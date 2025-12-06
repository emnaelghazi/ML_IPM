"""
MAFAULDA Predictive Maintenance - 1D CNN Training
Trains 1D Convolutional Neural Network on raw sensor signals
"""

import logging
import time
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.python.keras import layers, models, regularizers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. CNN training will be skipped.")

from config import Config
from data_loader import MAFAULDADataLoader
from utils import (
    load_csv_safe, save_pickle, plot_learning_curves,
    check_overfitting, format_duration, get_timestamp
)

warnings.filterwarnings('ignore')


class CNNDataGenerator(keras.utils.Sequence):
    """
    Data generator for loading raw sensor data in batches.
    Prevents loading entire dataset into memory.
    """
    
    def __init__(self, file_paths: List[Path], labels: np.ndarray,
                 batch_size: int, n_sensors: int, signal_length: int,
                 shuffle: bool = True):
        """
        Initialize data generator.
        
        Args:
            file_paths: List of CSV file paths
            labels: Array of labels
            batch_size: Batch size
            n_sensors: Number of sensor channels
            signal_length: Length of each signal
            shuffle: Whether to shuffle data
        """
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.n_sensors = n_sensors
        self.signal_length = signal_length
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.
        
        Args:
            index: Batch index
            
        Returns:
            Tuple of (batch_X, batch_y)
        """
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.file_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_X = np.zeros((len(batch_indices), self.signal_length, self.n_sensors))
        batch_y = np.zeros(len(batch_indices), dtype=int)
        
        for i, idx in enumerate(batch_indices):
            try:
                df = load_csv_safe(self.file_paths[idx])
                
                signal_data = df.iloc[:self.signal_length, :self.n_sensors].values
                
                if signal_data.shape[0] < self.signal_length:
                    pad_length = self.signal_length - signal_data.shape[0]
                    signal_data = np.vstack([signal_data, np.zeros((pad_length, self.n_sensors))])
                
                batch_X[i] = signal_data
                batch_y[i] = self.labels[idx]
                
            except Exception as e:
                logging.warning(f"Error loading {self.file_paths[idx]}: {e}")
                batch_X[i] = np.zeros((self.signal_length, self.n_sensors))
                batch_y[i] = self.labels[idx]
        
        return batch_X, batch_y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class CNN1DTrainer:
    """
    Trains 1D Convolutional Neural Network for fault classification.
    """
    
    def __init__(self, config: Config):
        """
        Initialize CNN trainer.
        
        Args:
            config: Configuration object
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN training")
        
        self.config = config
        self.cnn_config = config.models['cnn_1d']
        self.model = None
        self.history = None
        
        if config.gpu.enabled:
            self._setup_gpu()
        
        logging.info("Initialized CNN1DTrainer")
    
    def _setup_gpu(self):
        """Configure GPU settings for TensorFlow."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                if self.config.gpu.memory_growth:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                
                if self.config.gpu.device:
                    device_id = int(self.config.gpu.device.split(':')[-1])
                    tf.config.set_visible_devices(gpus[device_id], 'GPU')
                
                if self.config.gpu.mixed_precision:
                    policy = keras.mixed_precision.Policy('mixed_float16')
                    keras.mixed_precision.set_global_policy(policy)
                    logging.info("Mixed precision enabled")
                
                logging.info(f"GPU configured: {len(gpus)} device(s)")
                
            except RuntimeError as e:
                logging.error(f"GPU setup error: {e}")
        else:
            logging.warning("No GPUs detected. Using CPU.")
    
    def build_model(self) -> keras.Model:
        """
        Build 1D CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        arch = self.cnn_config.architecture
        train_config = self.cnn_config.training
        
        inputs = layers.Input(
            shape=(arch['input_length'], arch['n_sensors']),
            name='sensor_input'
        )
        
        x = inputs
        
        for i, conv_layer in enumerate(arch['conv_layers']):
            x = layers.Conv1D(
                filters=conv_layer['filters'],
                kernel_size=conv_layer['kernel_size'],
                strides=conv_layer['strides'],
                padding='same',
                activation=None,
                name=f'conv1d_{i+1}'
            )(x)
            
            if conv_layer.get('batch_norm', False):
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            x = layers.Activation(conv_layer['activation'], name=f'act_{i+1}')(x)
            
            if conv_layer.get('dropout', 0) > 0:
                x = layers.Dropout(conv_layer['dropout'], name=f'dropout_{i+1}')(x)
        
        if arch['global_pooling'] == 'avg':
            x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        else:
            x = layers.GlobalMaxPooling1D(name='global_pool')(x)
        
        for i, dense_layer in enumerate(arch['dense_layers']):
            x = layers.Dense(
                units=dense_layer['units'],
                activation=dense_layer['activation'],
                kernel_regularizer=regularizers.l2(dense_layer.get('l2_reg', 0)),
                name=f'dense_{i+1}'
            )(x)
            
            if dense_layer.get('dropout', 0) > 0:
                x = layers.Dropout(dense_layer['dropout'], name=f'dense_dropout_{i+1}')(x)
        
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='CNN1D_FaultClassifier')
        
        optimizer = keras.optimizers.Adam(learning_rate=train_config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary(print_fn=logging.info)
        
        return model
    
    def train(self, train_generator: CNNDataGenerator,
             val_generator: CNNDataGenerator) -> keras.Model:
        """
        Train the CNN model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            
        Returns:
            Trained model
        """
        logging.info("Training 1D CNN...")
        start_time = time.time()
        
        train_config = self.cnn_config.training
        
        self.model = self.build_model()
        
        callback_list = []
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=train_config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=train_config['reduce_lr_factor'],
            patience=train_config['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        checkpoint_path = self.config.paths.models / f"cnn_checkpoint_{get_timestamp()}.h5"
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callback_list.append(checkpoint)
        
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=train_config['epochs'],
            callbacks=callback_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        train_acc = self.history.history['accuracy'][-1]
        val_acc = self.history.history['val_accuracy'][-1]
        
        check_overfitting(train_acc, val_acc, self.config.training.overfitting_threshold)
        
        logging.info(f"CNN training completed in {format_duration(training_time)}")
        logging.info(f"Final train accuracy: {train_acc:.4f}")
        logging.info(f"Final validation accuracy: {val_acc:.4f}")
        
        self._plot_training_history()
        
        return self.model
    
    def _plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            return
        
        plot_learning_curves(
            self.history.history['loss'],
            self.history.history['val_loss'],
            metric_name='Loss',
            save_path=self.config.paths.performance / 'cnn_loss_curves.png'
        )
        
        plot_learning_curves(
            self.history.history['accuracy'],
            self.history.history['val_accuracy'],
            metric_name='Accuracy',
            save_path=self.config.paths.performance / 'cnn_accuracy_curves.png'
        )
    
    def save_model(self, save_path: Path):
        """
        Save trained model.
        
        Args:
            save_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        self.model.save(save_path)
        logging.info(f"Saved CNN model to {save_path}")
        
        if self.history is not None:
            history_path = save_path.parent / f"{save_path.stem}_history.pkl"
            save_pickle(self.history.history, history_path)
    
    def load_model(self, load_path: Path):
        """
        Load trained model.
        
        Args:
            load_path: Path to model file
        """
        self.model = keras.models.load_model(load_path)
        logging.info(f"Loaded CNN model from {load_path}")


def main():
    """Main CNN training function."""
    from config import get_config
    
    if not TENSORFLOW_AVAILABLE:
        logging.error("TensorFlow not available. Cannot train CNN.")
        return
    
    config = get_config()
    config.setup_reproducibility()
    config.setup_gpu()
    
    loader = MAFAULDADataLoader(config)
    splits_file = config.paths.splits / "data_splits.pkl"
    
    if not splits_file.exists():
        logging.error("Data splits not found. Run data_loader.py first.")
        return
    
    loader.load_splits()
    
    train_files = loader.get_split_files('train')
    val_files = loader.get_split_files('val')
    
    train_labels = np.array([loader.file_labels[str(f)] for f in train_files])
    val_labels = np.array([loader.file_labels[str(f)] for f in val_files])
    
    logging.info(f"Training files: {len(train_files)}")
    logging.info(f"Validation files: {len(val_files)}")
    
    cnn_config = config.models['cnn_1d']
    arch = cnn_config.architecture
    train_config = cnn_config.training
    
    train_generator = CNNDataGenerator(
        train_files, train_labels,
        batch_size=train_config['batch_size'],
        n_sensors=arch['n_sensors'],
        signal_length=arch['input_length'],
        shuffle=True
    )
    
    val_generator = CNNDataGenerator(
        val_files, val_labels,
        batch_size=train_config['batch_size'],
        n_sensors=arch['n_sensors'],
        signal_length=arch['input_length'],
        shuffle=False
    )
    
    # Initialize trainer
    trainer = CNN1DTrainer(config)
    
    model = trainer.train(train_generator, val_generator)
    
    model_path = config.paths.models / f"cnn_1d_{get_timestamp()}.h5"
    trainer.save_model(model_path)
    
    logging.info("\nCNN training complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()