"""
MAFAULDA Predictive Maintenance - Data Loader
Handles loading and stratified splitting of MAFAULDA dataset at FILE level
CRITICAL: Prevents data leakage by never splitting rows from same CSV
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import Config
from utils import (
    load_pickle, parse_fault_from_path, get_fault_label, load_csv_safe,
    save_json, load_json, save_pickle, get_timestamp
)


class MAFAULDADataLoader:
    """
    MAFAULDA Dataset Loader with File-Level Stratified Splitting.
    
    ANTI-OVERFITTING MEASURES:
    1. Split at CSV file level, NOT row level
    2. Stratified splitting maintains fault distribution
    3. Track file assignments to prevent leakage
    4. Validate splits before saving
    """
    
    def __init__(self, config: Config):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.raw_data_path = config.paths.raw_data
        self.splits_path = config.paths.splits
        
        # File tracking
        self.all_files: List[Path] = []
        self.file_labels: Dict[str, int] = {}
        self.file_fault_types: Dict[str, str] = {}
        self.file_severities: Dict[str, Optional[str]] = {}
        
        # Split assignments
        self.train_files: List[Path] = []
        self.val_files: List[Path] = []
        self.test_files: List[Path] = []
        
        # Statistics
        self.class_distribution: Dict[str, int] = {}
        
        logging.info(f"Initialized MAFAULDADataLoader with data path: {self.raw_data_path}")
    
    def discover_files(self) -> None:
        """
        Discover all CSV files in the raw data directory.
        Parses fault types and labels from directory structure.
        """
        logging.info("Discovering CSV files in dataset...")
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path does not exist: {self.raw_data_path}")
        
        # Find all CSV files recursively
        self.all_files = sorted(list(self.raw_data_path.rglob("*.csv")))
        
        if not self.all_files:
            raise ValueError(f"No CSV files found in {self.raw_data_path}")
        
        logging.info(f"Found {len(self.all_files)} CSV files")
        
        # Parse fault information from paths
        for file_path in tqdm(self.all_files, desc="Parsing file paths"):
            try:
                # Extract fault type and severity
                fault_type, severity = parse_fault_from_path(file_path)
                
                # Get integer label
                label = get_fault_label(file_path, self.config.fault_classes)
                
                # Store information
                file_key = str(file_path)
                self.file_labels[file_key] = label
                self.file_fault_types[file_key] = fault_type
                self.file_severities[file_key] = severity
                
            except Exception as e:
                logging.warning(f"Could not parse {file_path}: {e}")
                continue
        
        # Calculate class distribution
        label_counts = Counter(self.file_labels.values())
        for label, count in label_counts.items():
            fault_name = self.config.fault_class_names[label]
            self.class_distribution[fault_name] = count
        
        logging.info(f"Class distribution (file-level):")
        for fault_name, count in sorted(self.class_distribution.items()):
            logging.info(f"  {fault_name}: {count} files")
        
        # Validate minimum samples per class
        min_samples = self.config.data.min_samples_per_class
        for fault_name, count in self.class_distribution.items():
            if count < min_samples:
                logging.warning(
                    f"Class '{fault_name}' has only {count} files "
                    f"(minimum: {min_samples})"
                )
    
    def create_stratified_splits(self, random_seed: Optional[int] = None) -> None:
        """
        Create stratified train/val/test splits at FILE level.
        
        CRITICAL: This ensures no rows from the same CSV appear in multiple splits.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if not self.all_files:
            raise ValueError("No files discovered. Run discover_files() first.")
        
        seed = random_seed if random_seed is not None else self.config.random_seed
        
        logging.info(f"Creating stratified splits with seed={seed}")
        logging.info(f"Split ratios: Train={self.config.data.train_ratio}, "
                    f"Val={self.config.data.val_ratio}, Test={self.config.data.test_ratio}")
        
        # Prepare data for stratification
        file_paths = []
        labels = []
        
        for file_path in self.all_files:
            file_key = str(file_path)
            if file_key in self.file_labels:
                file_paths.append(file_path)
                labels.append(self.file_labels[file_key])
        
        file_paths = np.array(file_paths)
        labels = np.array(labels)
        
        logging.info(f"Splitting {len(file_paths)} files across {len(np.unique(labels))} classes")
        
        # First split: train vs (val + test)
        train_size = self.config.data.train_ratio
        temp_size = self.config.data.val_ratio + self.config.data.test_ratio
        
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            file_paths, labels,
            train_size=train_size,
            stratify=labels,
            random_state=seed
        )
        
        # Second split: val vs test
        val_ratio = self.config.data.val_ratio / temp_size
        
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels,
            train_size=val_ratio,
            stratify=temp_labels,
            random_state=seed
        )
        
        # Store splits
        self.train_files = list(train_files)
        self.val_files = list(val_files)
        self.test_files = list(test_files)
        
        # Log split statistics
        self._log_split_statistics()
        
        # Validate splits
        self._validate_splits()
        
        logging.info("Stratified splits created successfully")
    
    def _log_split_statistics(self) -> None:
        """Log detailed statistics about the splits."""
        logging.info("\n" + "="*70)
        logging.info("SPLIT STATISTICS")
        logging.info("="*70)
        
        splits = {
            'Train': self.train_files,
            'Validation': self.val_files,
            'Test': self.test_files
        }
        
        for split_name, files in splits.items():
            logging.info(f"\n{split_name} Set: {len(files)} files")
            
            # Count labels in this split
            split_labels = [self.file_labels[str(f)] for f in files]
            label_counts = Counter(split_labels)
            
            # Show distribution
            for label in sorted(label_counts.keys()):
                fault_name = self.config.fault_class_names[label]
                count = label_counts[label]
                percentage = 100 * count / len(files)
                logging.info(f"  {fault_name}: {count} files ({percentage:.1f}%)")
        
        logging.info("="*70 + "\n")
    
    def _validate_splits(self) -> None:
        """
        Validate that splits are valid and no data leakage exists.
        
        Raises:
            ValueError: If validation fails
        """
        logging.info("Validating splits for data leakage...")
        
        # Check no overlap between splits
        train_set = set(str(f) for f in self.train_files)
        val_set = set(str(f) for f in self.val_files)
        test_set = set(str(f) for f in self.test_files)
        
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap:
            raise ValueError(f"Data leakage: {len(train_val_overlap)} files in both train and val")
        
        if train_test_overlap:
            raise ValueError(f"Data leakage: {len(train_test_overlap)} files in both train and test")
        
        if val_test_overlap:
            raise ValueError(f"Data leakage: {len(val_test_overlap)} files in both val and test")
        
        # Check all files are assigned
        total_assigned = len(self.train_files) + len(self.val_files) + len(self.test_files)
        total_files = len([f for f in self.all_files if str(f) in self.file_labels])
        
        if total_assigned != total_files:
            raise ValueError(
                f"File assignment mismatch: {total_assigned} assigned vs {total_files} total"
            )
        
        # Check each class is represented in each split
        for split_name, files in [('Train', self.train_files), 
                                   ('Validation', self.val_files),
                                   ('Test', self.test_files)]:
            split_labels = set(self.file_labels[str(f)] for f in files)
            all_labels = set(self.file_labels.values())
            
            missing_labels = all_labels - split_labels
            if missing_labels:
                missing_names = [self.config.fault_class_names[l] for l in missing_labels]
                logging.warning(
                    f"{split_name} split missing classes: {missing_names}"
                )
        
        logging.info("✓ Validation passed: No data leakage detected")
    
    def save_splits(self, suffix: str = "") -> None:
        """
        Save split assignments to disk.
        
        Args:
            suffix: Optional suffix for split files
        """
        self.splits_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = get_timestamp()
        suffix_str = f"_{suffix}" if suffix else ""
        
        # Save as JSON for human readability
        splits_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_files': len(self.all_files),
                'train_ratio': self.config.data.train_ratio,
                'val_ratio': self.config.data.val_ratio,
                'test_ratio': self.config.data.test_ratio,
                'random_seed': self.config.random_seed,
                'class_distribution': self.class_distribution
            },
            'train': [str(f) for f in self.train_files],
            'validation': [str(f) for f in self.val_files],
            'test': [str(f) for f in self.test_files],
            'file_labels': self.file_labels,
            'file_fault_types': self.file_fault_types,
            'file_severities': self.file_severities
        }
        
        json_path = self.splits_path / f"data_splits{suffix_str}.json"
        save_json(splits_data, json_path)
        
        # Also save as pickle for faster loading
        pickle_path = self.splits_path / f"data_splits{suffix_str}.pkl"
        save_pickle(splits_data, pickle_path)
        
        logging.info(f"Splits saved to {self.splits_path}")
    
    def load_splits(self, splits_file: Optional[str] = None) -> None:
        """
        Load previously saved splits.
        
        Args:
            splits_file: Path to splits file (uses most recent if not specified)
        """
        if splits_file is None:
            split_files = sorted(self.splits_path.glob("data_splits*.pkl"))
            if not split_files:
                raise FileNotFoundError(f"No split files found in {self.splits_path}")
            splits_file = split_files[-1]
        
        logging.info(f"Loading splits from {splits_file}")
        
        splits_data = load_pickle(splits_file)
        
        self.train_files = [Path(f) for f in splits_data['train']]
        self.val_files = [Path(f) for f in splits_data['validation']]
        self.test_files = [Path(f) for f in splits_data['test']]
        self.file_labels = splits_data['file_labels']
        self.file_fault_types = splits_data['file_fault_types']
        self.file_severities = splits_data['file_severities']
        
        # Reconstruct all_files
        self.all_files = self.train_files + self.val_files + self.test_files
        
        # Restore class distribution
        if 'metadata' in splits_data:
            self.class_distribution = splits_data['metadata'].get('class_distribution', {})
        
        logging.info(f"Loaded splits: {len(self.train_files)} train, "
                    f"{len(self.val_files)} val, {len(self.test_files)} test")
    
    def get_split_files(self, split: str) -> List[Path]:
        """
        Get file paths for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            List of file paths
        """
        split = split.lower()
        
        if split in ['train', 'training']:
            return self.train_files
        elif split in ['val', 'validation', 'valid']:
            return self.val_files
        elif split == 'test':
            return self.test_files
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def load_split_data(self, split: str, 
                       max_files: Optional[int] = None,
                       chunk_size: Optional[int] = None) -> Tuple[List[pd.DataFrame], np.ndarray]:
        """
        Load data for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
            max_files: Maximum number of files to load (for testing)
            chunk_size: Chunk size for loading large CSVs
            
        Returns:
            Tuple of (list of DataFrames, array of labels)
        """
        files = self.get_split_files(split)
        
        if max_files:
            files = files[:max_files]
        
        logging.info(f"Loading {len(files)} files from {split} split...")
        
        dataframes = []
        labels = []
        
        chunk_size = chunk_size or self.config.feature_engineering.chunk_size
        
        for file_path in tqdm(files, desc=f"Loading {split} data"):
            try:
                # Load CSV
                df = load_csv_safe(file_path, chunk_size=chunk_size)
                
                # Get label
                label = self.file_labels[str(file_path)]
                
                dataframes.append(df)
                labels.append(label)
                
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                continue
        
        labels = np.array(labels)
        
        logging.info(f"Loaded {len(dataframes)} DataFrames from {split} split")
        
        return dataframes, labels
    
    def get_class_weights(self, split: str = 'train') -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            split: Split to calculate weights for
            
        Returns:
            Dictionary mapping class labels to weights
        """
        files = self.get_split_files(split)
        labels = [self.file_labels[str(f)] for f in files]
        
        label_counts = Counter(labels)
        n_samples = len(labels)
        n_classes = len(label_counts)
        
        # Calculate balanced weights
        weights = {}
        for label, count in label_counts.items():
            weights[label] = n_samples / (n_classes * count)
        
        return weights
    
    def generate_data_report(self, save_path: Optional[Path] = None) -> Dict:
        """
        Generate comprehensive data report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report dictionary
        """
        report = {
            'dataset_info': {
                'total_files': len(self.all_files),
                'total_classes': len(self.class_distribution),
                'class_distribution': self.class_distribution
            },
            'splits': {
                'train': {
                    'num_files': len(self.train_files),
                    'percentage': 100 * len(self.train_files) / len(self.all_files)
                },
                'validation': {
                    'num_files': len(self.val_files),
                    'percentage': 100 * len(self.val_files) / len(self.all_files)
                },
                'test': {
                    'num_files': len(self.test_files),
                    'percentage': 100 * len(self.test_files) / len(self.all_files)
                }
            },
            'class_weights': self.get_class_weights('train')
        }
        
        if save_path:
            save_json(report, save_path)
        
        return report


def main():
    """Main function for testing data loader."""
    from config import get_config
    
    # Load configuration
    config = get_config()
    config.setup_reproducibility()
    config.create_directories()
    
    # Initialize data loader
    loader = MAFAULDADataLoader(config)
    
    # Discover files
    loader.discover_files()
    
    # Create splits
    loader.create_stratified_splits()
    
    # Save splits
    loader.save_splits()
    
    # Generate report
    report = loader.generate_data_report(
        save_path=config.paths.splits / "data_report.json"
    )
    
    logging.info("\nData loading pipeline complete!")
    logging.info(f"Report: {report}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()