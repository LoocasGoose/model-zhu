"""
python hyperparameter_tune_windows.py --cfg=configs/lenet_base.yaml --n-trials 30 --tune-epochs 6 --study-name lenet_windows_tuning

Hyperparameter tuning script optimized for Windows systems.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.utils.metrics import AverageMeter, accuracy
import copy
from tqdm import tqdm

from config import get_config
from data import build_loader
from models import build_model
from optimizer import build_optimizer


def parse_option():
    parser = argparse.ArgumentParser("Windows-friendly hyperparameter tuning", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of trials for hyperparameter search")
    parser.add_argument("--tune-epochs", type=int, default=6, help="Number of epochs for each trial")
    parser.add_argument("--study-name", type=str, default="lenet_windows_tuning", help="Name of the study")
    
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    return args, config


# Rest of the functions (train_one_epoch, validate) remain the same...


def objective(trial, config, dataset_train, dataset_val, tune_epochs):
    # Expanded parameter space with better ranges
    lr = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.85, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Create a copy of the config and make it mutable
    mutable_config = copy.deepcopy(config)
    mutable_config.defrost()
    
    # Modify config
    mutable_config.TRAIN.LR = lr
    mutable_config.TRAIN.OPTIMIZER.MOMENTUM = momentum
    mutable_config.TRAIN.OPTIMIZER.WEIGHT_DECAY = weight_decay
    mutable_config.DATA.BATCH_SIZE = batch_size
    mutable_config.DATA.NUM_WORKERS = 0  # Critical for Windows: avoid multiprocessing in data loaders
    
    # Freeze the config
    mutable_config.freeze()
    
    print(f"\nTrial {trial.number}: LR={lr:.6f}, Momentum={momentum:.3f}, Batch Size={batch_size}, "
          f"Weight Decay={weight_decay:.6f}, Activation={activation}")
    
    try:
        # Make sure GPU is clean before starting
        torch.cuda.empty_cache()
        
        # Create data loaders with the new batch size - IMPORTANT: num_workers=0 for Windows
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # No additional workers on Windows to avoid multiprocessing issues
            pin_memory=True
        )
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # No additional workers
            pin_memory=True
        )
        
        # Build model and continue as usual...
        
        # ... rest of function remains the same
    
    except Exception as e:
        print(f"Trial failed: {str(e)}")
        torch.cuda.empty_cache()
        return float('-inf')


def main():
    args, config = parse_option()
    tune_epochs = args.tune_epochs
    
    # Create output directory
    os.makedirs("optuna_results", exist_ok=True)
    study_name = args.study_name
    
    print("Loading datasets...")
    # Only load datasets once to save time
    dataset_train, dataset_val, _, _, _, _ = build_loader(config)
    
    # Print CUDA info
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Create Optuna study with better pruning
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=tune_epochs, reduction_factor=3
        )
    )
    
    print(f"Starting optimization with {args.n_trials} trials, {tune_epochs} epochs each...")
    
    # IMPORTANT: Run sequentially on Windows (n_jobs=1)
    study.optimize(
        lambda trial: objective(trial, config, dataset_train, dataset_val, tune_epochs),
        n_trials=args.n_trials,
        n_jobs=1  # Sequential execution to avoid Windows multiprocessing issues
    )
    
    # ... rest of function remains the same 