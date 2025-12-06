"""
MAFAULDA Predictive Maintenance - Feature Engineering
Extracts time-domain, frequency-domain, and time-frequency features
with parallel processing and memory efficiency
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats, signal, fft
from scipy.stats import kurtosis, skew
import pywt
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from joblib import Parallel, delayed
from tqdm import tqdm

from config import Config
from utils import load_csv_safe, save_pickle, load_pickle, get_timestamp

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extract features from sensor signals.
    Handles time-domain, frequency-domain, and time-frequency features.
    """
    
    def __init__(self, config: Config):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.fe_config = config.feature_engineering
        self.sensor_names = config.data.sensor_names
        self.n_sensors = config.data.sensor_columns
        
    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features from a signal.
        
        Args:
            signal: 1D numpy array of sensor data
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        if not self.fe_config.time_domain['enabled']:
            return features
        
        # Basic statistical features
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(signal)
        features['iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)
        
        # Shape features
        features['skewness'] = skew(signal)
        features['kurtosis'] = kurtosis(signal)
        
        # Energy-based features
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['peak_to_peak'] = np.ptp(signal)
        
        # Advanced features
        if features['rms'] > 1e-10:
            features['crest_factor'] = features['max'] / features['rms']
            features['clearance_factor'] = features['max'] / (np.mean(np.sqrt(np.abs(signal)))**2)
            features['shape_factor'] = features['rms'] / np.mean(np.abs(signal))
            features['impulse_factor'] = features['max'] / np.mean(np.abs(signal))
        else:
            features['crest_factor'] = 0
            features['clearance_factor'] = 0
            features['shape_factor'] = 0
            features['impulse_factor'] = 0
        
        # Percentiles
        features['percentile_25'] = np.percentile(signal, 25)
        features['percentile_75'] = np.percentile(signal, 75)
        
        # Zero crossing rate
        features['zero_crossing_rate'] = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
        
        return features
    
    def extract_frequency_domain_features(self, signal: np.ndarray, 
                                         sensor_idx: int = 0) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT.
        
        Args:
            signal: 1D numpy array of sensor data
            sensor_idx: Sensor index (for RPM calculation if tachometer)
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        if not self.fe_config.frequency_domain['enabled']:
            return features
        
        # Compute FFT
        n = len(signal)
        fft_values = fft.rfft(signal)
        fft_magnitude = np.abs(fft_values)
        fft_freq = fft.rfftfreq(n, d=1.0/self.fe_config.frequency_domain['sampling_rate'])
        
        # Normalize by signal length
        fft_magnitude = fft_magnitude / n
        
        # Dominant frequency and amplitude
        dominant_idx = np.argmax(fft_magnitude[1:]) + 1  # Skip DC component
        features['dominant_frequency'] = fft_freq[dominant_idx]
        features['dominant_amplitude'] = fft_magnitude[dominant_idx]
        
        # Spectral moments
        power_spectrum = fft_magnitude ** 2
        total_power = np.sum(power_spectrum)
        
        if total_power > 1e-10:
            # Spectral centroid (center of mass)
            features['spectral_centroid'] = np.sum(fft_freq * power_spectrum) / total_power
            
            # Spectral spread (standard deviation)
            features['spectral_spread'] = np.sqrt(
                np.sum(((fft_freq - features['spectral_centroid'])**2) * power_spectrum) / total_power
            )
            
            # Spectral rolloff (95% of power)
            cumsum_power = np.cumsum(power_spectrum)
            rolloff_idx = np.where(cumsum_power >= 0.95 * total_power)[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = fft_freq[rolloff_idx[0]]
            else:
                features['spectral_rolloff'] = fft_freq[-1]
            
            # Spectral entropy
            power_norm = power_spectrum / total_power
            power_norm = power_norm[power_norm > 1e-10]
            features['spectral_entropy'] = -np.sum(power_norm * np.log2(power_norm))
        else:
            features['spectral_centroid'] = 0
            features['spectral_spread'] = 0
            features['spectral_rolloff'] = 0
            features['spectral_entropy'] = 0
        
        # Power in frequency bands
        low_freq_mask = fft_freq < 1000
        mid_freq_mask = (fft_freq >= 1000) & (fft_freq < 5000)
        high_freq_mask = fft_freq >= 5000
        
        features['power_low_freq'] = np.sum(power_spectrum[low_freq_mask])
        features['power_mid_freq'] = np.sum(power_spectrum[mid_freq_mask])
        features['power_high_freq'] = np.sum(power_spectrum[high_freq_mask])
        
        # Peak frequencies (top 3)
        top_3_idx = np.argsort(fft_magnitude[1:])[-3:][::-1] + 1
        for i, idx in enumerate(top_3_idx):
            features[f'peak_freq_{i+1}'] = fft_freq[idx]
            features[f'peak_amplitude_{i+1}'] = fft_magnitude[idx]
        
        # Bearing characteristic frequencies (if enabled and RPM available)
        if (self.fe_config.frequency_domain['bearing_frequencies']['enabled'] 
            and sensor_idx == 0):  # Tachometer
            rpm = self._estimate_rpm(signal)
            if rpm > 0:
                bcf_features = self._calculate_bearing_frequencies(rpm, fft_freq, fft_magnitude)
                features.update(bcf_features)
        
        return features
    
    def extract_time_frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time-frequency features using wavelet transform.
        
        Args:
            signal: 1D numpy array of sensor data
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        if not self.fe_config.time_frequency['enabled']:
            return features
        
        wavelet = self.fe_config.time_frequency['wavelet_type']
        level = self.fe_config.time_frequency['decomposition_level']
        
        try:
            # Wavelet decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            
            # Features from each decomposition level
            for i, coeff in enumerate(coeffs):
                prefix = f'wavelet_level_{i}'
                
                # Energy
                features[f'{prefix}_energy'] = np.sum(coeff**2)
                
                # Standard deviation
                features[f'{prefix}_std'] = np.std(coeff)
                
                # Mean absolute value
                features[f'{prefix}_mean_abs'] = np.mean(np.abs(coeff))
                
                # Entropy
                coeff_norm = coeff**2
                coeff_norm = coeff_norm / (np.sum(coeff_norm) + 1e-10)
                coeff_norm = coeff_norm[coeff_norm > 1e-10]
                features[f'{prefix}_entropy'] = -np.sum(coeff_norm * np.log2(coeff_norm))
            
            # Total wavelet energy
            features['wavelet_total_energy'] = sum(np.sum(c**2) for c in coeffs)
            
        except Exception as e:
            logging.warning(f"Wavelet decomposition failed: {e}")
            # Return zero features
            for i in range(level + 1):
                prefix = f'wavelet_level_{i}'
                features[f'{prefix}_energy'] = 0
                features[f'{prefix}_std'] = 0
                features[f'{prefix}_mean_abs'] = 0
                features[f'{prefix}_entropy'] = 0
            features['wavelet_total_energy'] = 0
        
        return features
    
    def _estimate_rpm(self, tachometer_signal: np.ndarray) -> float:
        """
        Estimate RPM from tachometer signal.
        
        Args:
            tachometer_signal: Tachometer sensor data
            
        Returns:
            Estimated RPM
        """
        try:
            # Find peaks in tachometer signal
            peaks, _ = signal.find_peaks(tachometer_signal, height=np.mean(tachometer_signal))
            
            if len(peaks) < 2:
                return 0
            
            # Calculate average time between peaks
            sampling_rate = self.fe_config.frequency_domain['sampling_rate']
            peak_intervals = np.diff(peaks) / sampling_rate  # seconds
            avg_interval = np.mean(peak_intervals)
            
            # Convert to RPM
            rpm = 60.0 / avg_interval if avg_interval > 0 else 0
            
            return rpm
        except:
            return 0
    
    def _calculate_bearing_frequencies(self, rpm: float, fft_freq: np.ndarray, 
                                      fft_magnitude: np.ndarray) -> Dict[str, float]:
        """
        Calculate bearing characteristic frequencies and extract amplitudes.
        
        Args:
            rpm: Rotational speed in RPM
            fft_freq: FFT frequency array
            fft_magnitude: FFT magnitude array
            
        Returns:
            Dictionary of bearing frequency features
        """
        features = {}
        
        bf_config = self.fe_config.frequency_domain['bearing_frequencies']
        
        # Bearing parameters
        N = bf_config['num_balls']
        d = bf_config['ball_diameter']
        D = bf_config['pitch_diameter']
        phi = np.radians(bf_config['contact_angle'])
        
        # Calculate characteristic frequencies
        rps = rpm / 60.0  # Revolutions per second
        
        # Ball Pass Frequency Outer (BPFO)
        bpfo = (N / 2) * (1 + (d / D) * np.cos(phi)) * rps
        
        # Ball Pass Frequency Inner (BPFI)
        bpfi = (N / 2) * (1 - (d / D) * np.cos(phi)) * rps
        
        # Ball Spin Frequency (BSF)
        bsf = (D / (2 * d)) * (1 - ((d / D) * np.cos(phi))**2) * rps
        
        # Fundamental Train Frequency (FTF)
        ftf = (1 / 2) * (1 - (d / D) * np.cos(phi)) * rps
        
        # Extract amplitudes at these frequencies (with tolerance)
        tolerance = bf_config['tolerance']
        
        for freq_name, freq_value in [('bpfo', bpfo), ('bpfi', bpfi), 
                                       ('bsf', bsf), ('ftf', ftf)]:
            # Find nearest frequency bin
            mask = np.abs(fft_freq - freq_value) <= (freq_value * tolerance)
            if np.any(mask):
                features[f'{freq_name}_amplitude'] = np.max(fft_magnitude[mask])
            else:
                features[f'{freq_name}_amplitude'] = 0
            
            features[f'{freq_name}_frequency'] = freq_value
        
        return features
    
    def extract_all_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all features from all sensors in a DataFrame.
        
        Args:
            df: DataFrame with sensor columns
            
        Returns:
            Dictionary of all features
        """
        all_features = {}
        
        # Process each sensor
        for sensor_idx in range(min(self.n_sensors, df.shape[1])):
            sensor_name = self.sensor_names[sensor_idx]
            signal = df.iloc[:, sensor_idx].values
            
            # Time-domain features
            time_features = self.extract_time_domain_features(signal)
            for feat_name, feat_value in time_features.items():
                all_features[f'{sensor_name}_{feat_name}'] = feat_value
            
            # Frequency-domain features
            freq_features = self.extract_frequency_domain_features(signal, sensor_idx)
            for feat_name, feat_value in freq_features.items():
                all_features[f'{sensor_name}_{feat_name}'] = feat_value
            
            # Time-frequency features
            tf_features = self.extract_time_frequency_features(signal)
            for feat_name, feat_value in tf_features.items():
                all_features[f'{sensor_name}_{feat_name}'] = feat_value
        
        return all_features


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline with fitting and transformation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.extractor = FeatureExtractor(config)
        
        # Scaler
        self.scaler = None
        self.scaling_method = config.feature_engineering.scaling_method
        
        # Feature selection
        self.feature_selector = None
        self.selected_features = None
        
        # Fitted flag
        self.is_fitted = False
        
        logging.info("Initialized FeatureEngineeringPipeline")
    
    def _process_single_file(self, file_path: Path) -> Optional[Dict[str, float]]:
        """
        Process a single CSV file and extract features.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dictionary of features or None if error
        """
        try:
            # Load CSV
            df = load_csv_safe(file_path, chunk_size=self.config.feature_engineering.chunk_size)
            
            # Extract features
            features = self.extractor.extract_all_features(df)
            
            return features
        
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None
    
    def extract_features_parallel(self, file_paths: List[Path], 
                                  desc: str = "Extracting features") -> pd.DataFrame:
        """
        Extract features from multiple files in parallel.
        
        Args:
            file_paths: List of CSV file paths
            desc: Progress bar description
            
        Returns:
            DataFrame with features
        """
        n_jobs = self.config.feature_engineering.n_jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        logging.info(f"Extracting features from {len(file_paths)} files using {n_jobs} cores")
        
        # Process in batches to manage memory
        batch_size = self.config.feature_engineering.batch_size
        all_features = []
        
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]
            
            # Parallel processing
            batch_features = Parallel(n_jobs=n_jobs)(
                delayed(self._process_single_file)(file_path)
                for file_path in tqdm(batch_files, desc=f"{desc} (batch {i//batch_size + 1})")
            )
            
            # Filter out None results
            batch_features = [f for f in batch_features if f is not None]
            all_features.extend(batch_features)
            
            logging.info(f"Processed batch {i//batch_size + 1}, total features: {len(all_features)}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        logging.info(f"Extracted {features_df.shape[1]} features from {features_df.shape[0]} files")
        
        return features_df
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FeatureEngineeringPipeline':
        """
        Fit the feature engineering pipeline on training data.
        
        Args:
            X: Feature matrix (training data)
            y: Labels (training data)
            
        Returns:
            Self
        """
        logging.info("Fitting feature engineering pipeline...")
        
        # Remove features with zero variance
        variance = X.var()
        non_zero_var_features = variance[variance > 1e-10].index
        X_filtered = X[non_zero_var_features]
        
        logging.info(f"Removed {len(X.columns) - len(non_zero_var_features)} zero-variance features")
        
        # Remove highly correlated features
        if self.config.feature_engineering.feature_selection['enabled']:
            correlation_matrix = X_filtered.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            threshold = self.config.feature_engineering.feature_selection['correlation_threshold']
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]
            
            X_filtered = X_filtered.drop(columns=to_drop)
            
            logging.info(f"Removed {len(to_drop)} highly correlated features (threshold={threshold})")
        
        # Feature selection
        if self.config.feature_engineering.feature_selection['enabled']:
            n_features = self.config.feature_engineering.feature_selection['n_features']
            n_features = min(n_features, X_filtered.shape[1])
            
            method = self.config.feature_engineering.feature_selection['method']
            
            if method == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=n_features)
            elif method == 'f_classif':
                selector = SelectKBest(f_classif, k=n_features)
            else:
                selector = SelectKBest(f_classif, k=n_features)
            
            selector.fit(X_filtered, y)
            selected_mask = selector.get_support()
            self.selected_features = X_filtered.columns[selected_mask].tolist()
            
            X_filtered = X_filtered[self.selected_features]
            
            logging.info(f"Selected {len(self.selected_features)} best features using {method}")
        else:
            self.selected_features = X_filtered.columns.tolist()
        
        # Fit scaler
        if self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
        
        self.scaler.fit(X_filtered)
        
        logging.info(f"Fitted {self.scaling_method} scaler")
        
        self.is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Select features
        X_selected = X[self.selected_features]
        
        # Scale
        X_scaled = self.scaler.transform(X_selected)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Transformed feature matrix
        """
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, save_path: Path) -> None:
        """
        Save fitted pipeline.
        
        Args:
            save_path: Path to save pipeline
        """
        pipeline_data = {
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'scaling_method': self.scaling_method,
            'is_fitted': self.is_fitted
        }
        
        save_pickle(pipeline_data, save_path)
        logging.info(f"Saved feature engineering pipeline to {save_path}")
    
    def load(self, load_path: Path) -> None:
        """
        Load fitted pipeline.
        
        Args:
            load_path: Path to load pipeline from
        """
        pipeline_data = load_pickle(load_path)
        
        self.scaler = pipeline_data['scaler']
        self.selected_features = pipeline_data['selected_features']
        self.scaling_method = pipeline_data['scaling_method']
        self.is_fitted = pipeline_data['is_fitted']
        
        logging.info(f"Loaded feature engineering pipeline from {load_path}")


def main():
    """Main function for feature engineering."""
    from config import get_config
    from data_loader import MAFAULDADataLoader
    
    # Setup
    config = get_config()
    config.setup_reproducibility()
    config.create_directories()
    
    # Load data splits
    loader = MAFAULDADataLoader(config)
    
    # Check if splits exist
    splits_file = config.paths.splits / "data_splits.pkl"
    if not splits_file.exists():
        logging.info("Creating data splits...")
        loader.discover_files()
        loader.create_stratified_splits()
        loader.save_splits()
    else:
        logging.info("Loading existing splits...")
        loader.load_splits()
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(config)
    
    # Extract features for each split
    for split_name in ['train', 'val', 'test']:
        logging.info(f"\n{'='*70}")
        logging.info(f"Processing {split_name.upper()} split")
        logging.info(f"{'='*70}")
        
        # Get files
        files = loader.get_split_files(split_name)
        labels = np.array([loader.file_labels[str(f)] for f in files])
        
        # Extract features
        features_df = pipeline.extract_features_parallel(files, desc=f"Extracting {split_name}")
        
        if split_name == 'train':
            logging.info("Fitting feature engineering pipeline...")
            features_scaled = pipeline.fit_transform(features_df, labels)
            
            pipeline.save(config.paths.scalers / "feature_pipeline.pkl")
        else:
            features_scaled = pipeline.transform(features_df)
        
        save_path = config.paths.processed_data / f"features_{split_name}.parquet"
        features_scaled_df = pd.DataFrame(
            features_scaled,
            columns=pipeline.selected_features
        )
        features_scaled_df['label'] = labels
        features_scaled_df.to_parquet(save_path, index=False)
        
        logging.info(f"Saved {split_name} features to {save_path}")
        logging.info(f"Shape: {features_scaled_df.shape}")
    
    logging.info("\nFeature engineering complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()