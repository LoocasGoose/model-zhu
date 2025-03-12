"""
hyperparameter_tune_alexnet.py
Script for tuning hyperparameters specifically for AlexNet.

CUDA_VISIBLE_DEVICES=5 python hyperparameter_tune_alexnet.py --cfg=configs/alexnet.yaml --n-trials 3 --tune-epochs 2 --n-jobs 1 --subset-ratio 0.05 --early-stop-patience 1 --study-name alexnet_tuning_test
"""

import argparse
import os
import json
import torch
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.utils.metrics import AverageMeter, accuracy
from tqdm import tqdm
import copy  # Added for deep copying config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import get_config
from data import build_loader
from models import build_model
from optimizer import build_optimizer

def parse_option():
    parser = argparse.ArgumentParser("Hyperparameter tuning for AlexNet", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials for hyperparameter search")
    parser.add_argument("--tune-epochs", type=int, default=8, help="Number of epochs for each trial")
    parser.add_argument("--study-name", type=str, default="alexnet_tuning", help="Name of the study")
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of parallel jobs for trials")
    parser.add_argument("--subset-ratio", type=float, default=0.1, 
                      help="Ratio of training data to use for hyperparameter tuning (0.0-1.0)")
    parser.add_argument("--early-stop-patience", type=int, default=2,
                      help="Number of epochs with no improvement after which to stop")
    
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    return args, config

def objective(trial, config, dataset_train, dataset_val, tune_epochs, subset_ratio=0.1, patience=2):
    # Define hyperparameters specific to AlexNet
    lr = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    # Log the parameters for the current trial
    logger.info(f"Trial {trial.number}: Learning Rate: {lr}, Batch Size: {batch_size}, Dropout Rate: {dropout_rate}")
    
    try:
        # Create a deep copy of the config (works with SimpleNamespace-like objects)
        mutable_config = copy.deepcopy(config)
        
        # Check if config has the defrost method (for some config objects)
        if hasattr(mutable_config, 'defrost'):
            mutable_config.defrost()
        
        # Modify config for tuning - safely set attributes
        if hasattr(mutable_config, 'TRAIN') and hasattr(mutable_config.TRAIN, 'LR'):
            mutable_config.TRAIN.LR = lr
        
        if hasattr(mutable_config, 'DATA') and hasattr(mutable_config.DATA, 'BATCH_SIZE'):
            mutable_config.DATA.BATCH_SIZE = batch_size
        
        if hasattr(mutable_config, 'MODEL') and hasattr(mutable_config.MODEL, 'DROP_RATE'):
            mutable_config.MODEL.DROP_RATE = dropout_rate
        
        # Freeze config if it has that method
        if hasattr(mutable_config, 'freeze'):
            mutable_config.freeze()
        
        # Build model
        model = build_model(mutable_config)
        
        # Use more training data for better representation
        train_size = int(len(dataset_train) * subset_ratio)
        indices = torch.randperm(len(dataset_train))[:train_size]
        subset_train = torch.utils.data.Subset(dataset_train, indices.tolist())
        
        # Use the full validation set for more accurate evaluation
        subset_val = dataset_val
        
        # Train and validate the model with the mutable config
        best_acc = train_and_validate(model, subset_train, subset_val, tune_epochs, batch_size, patience, mutable_config)
        
        # Log the best accuracy for the current trial
        logger.info(f"Trial {trial.number} completed with best accuracy: {best_acc:.2f}%")
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache()
        
        return best_acc
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        torch.cuda.empty_cache()
        return float('-inf')  # Return worst possible score

def train_and_validate(model, dataset_train, dataset_val, tune_epochs, batch_size, patience=2, config=None):
    logger.info("Starting training and validation...")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create data loaders with optimized parameters
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=batch_size*2,  # Double batch size for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Track epochs without improvement for early stopping
    epochs_no_improve = 0
    best_acc = 0.0
    
    # Setup optimizer and scheduler
    # Default values
    lr = 0.001
    min_lr = 1e-6
    weight_decay = 0.01
    momentum = 0.9
    betas = (0.9, 0.999)
    
    # Try to get values from config if available
    if config is not None:
        # Extract learning rate
        if hasattr(config, 'TRAIN') and hasattr(config.TRAIN, 'LR'):
            lr = config.TRAIN.LR
        
        # Extract min learning rate
        if hasattr(config, 'TRAIN') and hasattr(config.TRAIN, 'MIN_LR'):
            min_lr = config.TRAIN.MIN_LR
            
        # Get optimizer configuration if available
        if hasattr(config, 'TRAIN') and hasattr(config.TRAIN, 'OPTIMIZER'):
            optimizer_config = config.TRAIN.OPTIMIZER
            
            # Extract optimizer name with default
            optimizer_name = 'sgd'
            if hasattr(optimizer_config, 'NAME'):
                optimizer_name = optimizer_config.NAME.lower()
                
            # Extract weight decay if available
            if hasattr(optimizer_config, 'WEIGHT_DECAY'):
                weight_decay = optimizer_config.WEIGHT_DECAY
                
            # Extract momentum if available
            if hasattr(optimizer_config, 'MOMENTUM'):
                momentum = optimizer_config.MOMENTUM
                
            # Extract betas if available
            if hasattr(optimizer_config, 'BETAS'):
                betas = optimizer_config.BETAS
        else:
            optimizer_name = 'sgd'  # Default
    else:
        optimizer_name = 'sgd'  # Default
    
    # Create optimizer based on name
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
    else:  # Default to SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum
        )
    
    # Create learning rate scheduler
    lr_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=tune_epochs,
        eta_min=min_lr
    )
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(tune_epochs):
        logger.info(f"Epoch {epoch + 1}/{tune_epochs}")
        model.train()
        
        # Training with progress bar
        train_loss = 0.0
        with tqdm(total=len(data_loader_train), desc="Training", unit="batch") as pbar:
            for batch_idx, (images, targets) in enumerate(data_loader_train):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Zero gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        # Update learning rate
        lr_scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(total=len(data_loader_val), desc="Validation", unit="batch") as pbar:
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(data_loader_val):
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    
                    # Update metrics
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Update progress bar
                    pbar.update(1)
        
        # Calculate accuracy
        acc = 100.0 * correct / total
        logger.info(f"Validation Accuracy: {acc:.2f}%")
        
        # Check for improvement
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_acc

def main():
    args, config = parse_option()
    
    # Log the tuning parameters
    logger.info(f"Starting hyperparameter tuning for AlexNet with:")
    logger.info(f"  - Config file: {args.cfg}")
    logger.info(f"  - Number of trials: {args.n_trials}")
    logger.info(f"  - Epochs per trial: {args.tune_epochs}")
    logger.info(f"  - Parallel jobs: {args.n_jobs}")
    logger.info(f"  - Data subset ratio: {args.subset_ratio}")
    logger.info(f"  - Early stopping patience: {args.early_stop_patience}")
    
    # Build data loaders from config
    dataset_train, dataset_val, _, _, _, _ = build_loader(config)
    
    # Create output directory for results
    os.makedirs("optuna_results", exist_ok=True)
    
    # Create Optuna study with better pruning
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=args.tune_epochs, reduction_factor=3
        )
    )
    
    logger.info(f"Starting optimization with {args.n_trials} trials, {args.tune_epochs} epochs each...")
    logger.info(f"Using {args.subset_ratio*100:.1f}% of training data for faster tuning")
    logger.info(f"Running {args.n_jobs} trials in parallel!")
    
    study.optimize(
        lambda trial: objective(
            trial, 
            config, 
            dataset_train, 
            dataset_val, 
            args.tune_epochs, 
            args.subset_ratio,
            args.early_stop_patience
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs  # Run multiple trials in parallel
    )
    
    # Print results
    logger.info("\n==== Study Results ====")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    if study.best_trial:
        logger.info("\nBest trial:")
        trial = study.best_trial
        logger.info(f"  Value: {trial.value:.2f}%")
        logger.info("  Params:")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
        
        # Save best command for full training
        best_cmd = f"python main.py --cfg=configs/alexnet.yaml --opts"
        best_cmd += f" DATA.BATCH_SIZE {trial.params['batch_size']}"
        best_cmd += f" TRAIN.LR {trial.params['learning_rate']}"
        best_cmd += f" MODEL.DROP_RATE {trial.params['dropout_rate']}"
        best_cmd += f" TRAIN.EPOCHS 20"
        
        logger.info("\nRun this command with best parameters for full training:")
        logger.info(best_cmd)
        
        # Save results
        result_file = f"optuna_results/{args.study_name}_results.json"
        with open(result_file, "w") as f:
            json.dump({
                "best_trial": {
                    "value": trial.value,
                    "params": trial.params,
                    "command": best_cmd
                },
                "all_trials": [
                    {
                        "trial_id": t.number,
                        "value": t.value if t.value is not None else float('nan'),
                        "params": t.params,
                        "state": str(t.state)
                    }
                    for t in study.trials
                ]
            }, f, indent=2)
        
        logger.info(f"Results saved to {result_file}")
    else:
        logger.info("No successful trials completed")
    
    # Output the best parameters and accuracy
    logger.info("\nBest Parameters and Accuracy:")
    if study.best_trial:
        logger.info(f"Best Accuracy: {study.best_trial.value:.2f}%")
        for key, value in study.best_trial.params.items():
            logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main() 