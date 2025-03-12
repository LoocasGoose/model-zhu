"""
Main entry script for training models with YAML configuration files.

Example usage:
    python main.py --cfg configs/densenet.yaml
"""

import os
import sys
import argparse
import logging
import yaml
from types import SimpleNamespace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelTraining")


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        SimpleNamespace object with configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Convert dictionary to namespace for dot notation access
            return SimpleNamespace(**config)
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        sys.exit(1)


def create_directories(config):
    """
    Create necessary directories for checkpoints and logs
    
    Args:
        config: Configuration namespace
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    logger.info(f"Created directories: {config.checkpoint_dir}, {config.log_dir}")


def train(config):
    """
    Train the model based on the configuration
    
    Args:
        config: Configuration namespace
    """
    model_type = getattr(config, 'model_type', None)
    
    if model_type == 'densenet':
        # Import DenseNet training function
        try:
            from train_densenet import train_model
            logger.info("Starting DenseNet training...")
            train_model(config)
        except ImportError:
            logger.error("Failed to import DenseNet training module. Make sure train_densenet.py exists.")
            sys.exit(1)
    else:
        logger.error(f"Unsupported model type: {model_type}")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train models with YAML configuration")
    parser.add_argument("--cfg", required=True, help="Path to the configuration YAML file")
    
    args = parser.parse_args()
    
    # Check if configuration file exists
    if not os.path.exists(args.cfg):
        logger.error(f"Configuration file not found: {args.cfg}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.cfg)
    
    # Create necessary directories
    create_directories(config)
    
    # Train the model
    train(config)


if __name__ == "__main__":
    main()
