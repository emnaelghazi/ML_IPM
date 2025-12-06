"""
MAFAULDA Predictive Maintenance - Initial Setup Script
Run this script after installation to verify setup and prepare environment
"""

import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logging.error(f"Python 3.10+ required. You have {version.major}.{version.minor}")
        return False
    logging.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm',
        'matplotlib', 'seaborn', 'plotly', 'streamlit', 'pyyaml',
        'scipy', 'pywavelets', 'joblib', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logging.info(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            logging.warning(f"❌ {package} not found")
    
    if missing_packages:
        logging.error(f"\nMissing packages: {', '.join(missing_packages)}")
        logging.info("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_tensorflow():
    """Check TensorFlow installation and GPU availability."""
    try:
        import tensorflow as tf
        logging.info(f"✅ TensorFlow {tf.__version__} installed")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logging.info(f"✅ {len(gpus)} GPU(s) detected")
            for gpu in gpus:
                logging.info(f"   - {gpu.name}")
        else:
            logging.warning("⚠️  No GPUs detected (CPU mode)")
        
        return True
    except ImportError:
        logging.warning("⚠️  TensorFlow not installed (CNN training will be unavailable)")
        logging.info("Install with: pip install tensorflow[and-cuda]==2.15.0")
        return False


def check_directories():
    """Check and create required directories."""
    directories = [
        'data/raw',
        'data/processed',
        'data/splits',
        'models/trained_models',
        'models/scalers',
        'models/performance',
        'logs',
        'dashboard/uploads',
        'notebooks'
    ]
    
    project_root = Path(__file__).parent
    
    for dir_path in directories:
        full_path = project_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"✅ Created: {dir_path}")
        else:
            logging.info(f"✅ Exists: {dir_path}")
    
    return True


def check_data():
    """Check if MAFAULDA data is present."""
    data_path = Path(__file__).parent / 'data' / 'raw'
    
    csv_files = list(data_path.rglob('*.csv'))
    
    if not csv_files:
        logging.warning("⚠️  No CSV files found in data/raw/")
        logging.info("\nPlease place MAFAULDA dataset in data/raw/ with structure:")
        logging.info("  data/raw/normal/*.csv")
        logging.info("  data/raw/imbalance/6g/*.csv")
        logging.info("  data/raw/horizontal-misalignment/0.5mm/*.csv")
        logging.info("  etc.")
        return False
    
    logging.info(f"✅ Found {len(csv_files)} CSV files in data/raw/")
    return True


def check_config():
    """Check if config.yaml exists."""
    config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        logging.error("❌ config.yaml not found!")
        return False
    
    logging.info("✅ config.yaml found")
    
    # Try to load it
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info("✅ config.yaml is valid")
        return True
    except Exception as e:
        logging.error(f"❌ Error loading config.yaml: {e}")
        return False


def test_imports():
    """Test if src modules can be imported."""
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    modules = ['config', 'utils', 'data_loader']
    
    for module in modules:
        try:
            __import__(module)
            logging.info(f"✅ src.{module} can be imported")
        except Exception as e:
            logging.error(f"❌ Error importing src.{module}: {e}")
            return False
    
    return True


def print_next_steps():
    """Print next steps for user."""
    print("\n" + "="*70)
    print("SETUP COMPLETE - NEXT STEPS")
    print("="*70)
    print("\n1. Ensure MAFAULDA data is in data/raw/ directory")
    print("\n2. For GPU support (RTX 4050), see GPU_SETUP.md:")
    print("   pip install tensorflow[and-cuda]==2.15.0")
    print("\n3. Run the complete pipeline:")
    print("   python train_pipeline.py")
    print("\n4. Or run step-by-step:")
    print("   python src/data_loader.py          # Split data")
    print("   python src/feature_engineer.py     # Extract features")
    print("   python src/model_trainer.py        # Train models")
    print("   python src/model_predictor.py      # Evaluate")
    print("\n5. Launch dashboard:")
    print("   streamlit run dashboard/app.py")
    print("\n6. Explore notebooks:")
    print("   jupyter notebook notebooks/")
    print("\n" + "="*70)


def main():
    """Run all setup checks."""
    print("="*70)
    print("MAFAULDA PREDICTIVE MAINTENANCE - SETUP VERIFICATION")
    print("="*70)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("TensorFlow & GPU", check_tensorflow),
        ("Directory Structure", check_directories),
        ("Configuration File", check_config),
        ("Module Imports", test_imports),
        ("Data Availability", check_data)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{'='*70}")
        print(f"Checking: {check_name}")
        print(f"{'='*70}")
        result = check_func()
        results.append((check_name, result))
        print()
    
    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! System ready.")
        print_next_steps()
        return 0
    else:
        print("\n⚠️  Some checks failed. Please resolve issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())