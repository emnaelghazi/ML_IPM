"""
MAFAULDA Predictive Maintenance - Complete Training Pipeline
End-to-end pipeline from data loading to model evaluation
"""

import logging
import time
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import get_config
from data_loader import MAFAULDADataLoader
from feature_engineer import FeatureEngineeringPipeline
from model_trainer import ModelTrainer
from model_predictor import ModelPredictor
from utils import format_duration

import pandas as pd
import numpy as np


def setup_logging(config):
    """Setup logging configuration."""
    log_file = config.paths.logs / f"training_pipeline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging to {log_file}")


def step_1_data_preparation(config, force_resplit=False):
    """
    Step 1: Load and split data.
    
    Args:
        config: Configuration object
        force_resplit: Force recreation of splits
    """
    logging.info("\n" + "="*70)
    logging.info("STEP 1: DATA PREPARATION")
    logging.info("="*70)
    
    start_time = time.time()
    
    loader = MAFAULDADataLoader(config)
    splits_file = config.paths.splits / "data_splits.pkl"
    
    if splits_file.exists() and not force_resplit:
        logging.info("Loading existing data splits...")
        loader.load_splits()
    else:
        logging.info("Creating new data splits...")
        loader.discover_files()
        loader.create_stratified_splits()
        loader.save_splits()
        
        # Generate data report
        report = loader.generate_data_report(
            save_path=config.paths.splits / "data_report.json"
        )
        logging.info(f"Data report saved")
    
    duration = time.time() - start_time
    logging.info(f"Step 1 completed in {format_duration(duration)}")
    
    return loader


def step_2_feature_engineering(config, loader, force_reextract=False):
    """
    Step 2: Extract features from raw data.
    
    Args:
        config: Configuration object
        loader: Data loader with splits
        force_reextract: Force re-extraction of features
    """
    logging.info("\n" + "="*70)
    logging.info("STEP 2: FEATURE ENGINEERING")
    logging.info("="*70)
    
    start_time = time.time()
    
    # Check if features already exist
    features_exist = all([
        (config.paths.processed_data / f"features_{split}.parquet").exists()
        for split in ['train', 'val', 'test']
    ])
    
    if features_exist and not force_reextract:
        logging.info("Features already extracted. Loading from disk...")
        pipeline = FeatureEngineeringPipeline(config)
        pipeline.load(config.paths.scalers / "feature_pipeline.pkl")
    else:
        logging.info("Extracting features from raw data...")
        pipeline = FeatureEngineeringPipeline(config)
        
        # Extract features for each split
        for split_name in ['train', 'val', 'test']:
            logging.info(f"\nProcessing {split_name.upper()} split...")
            
            # Get files and labels
            files = loader.get_split_files(split_name)
            labels = np.array([loader.file_labels[str(f)] for f in files])
            
            # Extract features
            features_df = pipeline.extract_features_parallel(
                files, 
                desc=f"Extracting {split_name}"
            )
            
            # Fit/transform
            if split_name == 'train':
                features_scaled = pipeline.fit_transform(features_df, labels)
                pipeline.save(config.paths.scalers / "feature_pipeline.pkl")
            else:
                features_scaled = pipeline.transform(features_df)
            
            # Save features
            save_path = config.paths.processed_data / f"features_{split_name}.parquet"
            features_df_scaled = pd.DataFrame(
                features_scaled,
                columns=pipeline.selected_features
            )
            features_df_scaled['label'] = labels
            features_df_scaled.to_parquet(save_path, index=False)
            
            logging.info(f"Saved {split_name} features: {features_df_scaled.shape}")
    
    duration = time.time() - start_time
    logging.info(f"Step 2 completed in {format_duration(duration)}")
    
    return pipeline


def step_3_model_training(config):
    """
    Step 3: Train machine learning models.
    
    Args:
        config: Configuration object
    """
    logging.info("\n" + "="*70)
    logging.info("STEP 3: MODEL TRAINING")
    logging.info("="*70)
    
    start_time = time.time()
    
    logging.info("Loading preprocessed features...")
    train_df = pd.read_parquet(config.paths.processed_data / "features_train.parquet")
    val_df = pd.read_parquet(config.paths.processed_data / "features_val.parquet")
    
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
    
    duration = time.time() - start_time
    logging.info(f"Step 3 completed in {format_duration(duration)}")
    
    return trainer


def step_4_model_evaluation(config):
    """
    Step 4: Evaluate models on test set.
    
    Args:
        config: Configuration object
    """
    logging.info("\n" + "="*70)
    logging.info("STEP 4: MODEL EVALUATION")
    logging.info("="*70)
    
    start_time = time.time()
    
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
    
    # Evaluate all models
    results = predictor.evaluate_all_models(X_test, y_test, split_name='test')
    
    # Save results
    predictor.save_results(config.paths.performance)
    
    # Generate report
    report = predictor.generate_comprehensive_report()
    
    # Print summary
    logging.info(f"\n{'='*70}")
    logging.info("FINAL RESULTS")
    logging.info(f"{'='*70}")
    logging.info(f"Best Model: {report['best_model']['name']}")
    logging.info(f"Best F1-Macro: {report['best_model']['f1_macro']:.4f}")
    
    if 'business_metrics' in report:
        bm = report['business_metrics']
        logging.info(f"\nBusiness Metrics:")
        logging.info(f"  False Negative Rate: {bm['false_negative_rate']:.2%}")
        logging.info(f"  False Positive Rate: {bm['false_positive_rate']:.2%}")
    
    logging.info(f"{'='*70}\n")
    
    duration = time.time() - start_time
    logging.info(f"Step 4 completed in {format_duration(duration)}")
    
    return predictor, report


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='MAFAULDA Predictive Maintenance Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--force-resplit', action='store_true',
                       help='Force recreation of data splits')
    parser.add_argument('--force-reextract', action='store_true',
                       help='Force re-extraction of features')
    parser.add_argument('--skip-cnn', action='store_true',
                       help='Skip CNN training (faster)')
    parser.add_argument('--steps', type=str, default='all',
                       help='Steps to run (1,2,3,4 or "all")')
    
    args = parser.parse_args()
    
    # Parse steps
    if args.steps == 'all':
        steps_to_run = [1, 2, 3, 4]
    else:
        steps_to_run = [int(s) for s in args.steps.split(',')]
    
    # Load configuration
    config = get_config(args.config)
    
    # Setup
    setup_logging(config)
    config.setup_reproducibility()
    config.setup_gpu()
    config.create_directories()
    config.validate()
    
    # Log configuration
    logging.info("="*70)
    logging.info("MAFAULDA PREDICTIVE MAINTENANCE PIPELINE")
    logging.info("="*70)
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Random Seed: {config.random_seed}")
    logging.info(f"Enabled Models: {config.get_enabled_models()}")
    logging.info(f"GPU Enabled: {config.gpu.enabled}")
    logging.info(f"Steps to run: {steps_to_run}")
    logging.info("="*70)
    
    pipeline_start = time.time()
    
    try:
        # Step 1: Data Preparation
        if 1 in steps_to_run:
            loader = step_1_data_preparation(config, args.force_resplit)
        else:
            loader = MAFAULDADataLoader(config)
            loader.load_splits()
        
        # Step 2: Feature Engineering
        if 2 in steps_to_run:
            pipeline = step_2_feature_engineering(config, loader, args.force_reextract)
        
        # Step 3: Model Training
        if 3 in steps_to_run:
            trainer = step_3_model_training(config)
        
        # Step 4: Model Evaluation
        if 4 in steps_to_run:
            predictor, report = step_4_model_evaluation(config)
        
        # Total time
        total_time = time.time() - pipeline_start
        
        logging.info("\n" + "="*70)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("="*70)
        logging.info(f"Total Time: {format_duration(total_time)}")
        logging.info(f"Results saved to: {config.paths.performance}")
        logging.info("="*70)
        
        # Check if within time budget
        max_time = config.training.max_training_time_hours * 3600
        if total_time <= max_time:
            logging.info(f"✓ Completed within time budget ({total_time/3600:.2f}h / {config.training.max_training_time_hours}h)")
        else:
            logging.warning(f"⚠ Exceeded time budget ({total_time/3600:.2f}h / {config.training.max_training_time_hours}h)")
        
    except KeyboardInterrupt:
        logging.warning("\nPipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logging.error(f"\nPipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()