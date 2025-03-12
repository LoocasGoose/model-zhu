"""
hyperparameter_tune.py
~~~~~~~~~~~~~~~~~~~~~~

You can adjust these parameters:
--n-trials: Number of different hyperparameter combinations to try (default: 20)
--tune-epochs: Number of epochs to train each trial (default: 5)
--study-name: Name for your study (for saving results)

After running conda activate vision-zoo, run this script with:
python hyperparameter_tune.py --cfg=configs/lenet_base.yaml --opts --n-trials 20 --tune-epochs 5 --study-name "lenet_tuning"

pip dependencies: optuna

"""

import argparse
import datetime
import json
import os
import time
import numpy as np
import optuna
import torch
import torch.nn as nn
from timm.utils.metrics import AverageMeter, accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

from config import get_config
from data import build_loader
from models import build_model
from optimizer import build_optimizer
from utils import create_logger, load_checkpoint, save_checkpoint


def parse_option():
    parser = argparse.ArgumentParser("Hyperparameter tuning script", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials for hyperparameter search")
    parser.add_argument("--tune-epochs", type=int, default=5, help="Number of epochs for each trial")
    parser.add_argument("--study-name", type=str, default="lenet_tuning", help="Name of the study")
    
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    return args, config


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch):
    model.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        output = model(images)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        accs.update(acc1.item(), images.size(0))
        
    return losses.avg, accs.avg


def validate(config, data_loader, model):
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for idx, (images, target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            output = model(images)
            loss = criterion(output, target)
            
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            accs.update(acc1.item(), images.size(0))
            
    return losses.avg, accs.avg


def objective(trial, config, data_loader_train, data_loader_val, tune_epochs):
    # Define the hyperparameters to tune with more conservative batch sizes
    lr = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    # Use smaller batch sizes to avoid CUDA OOM errors
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])
    
    # Create a copy of the config and make it mutable
    mutable_config = copy.deepcopy(config)
    mutable_config.defrost()
    
    # Now we can modify the config
    mutable_config.TRAIN.LR = lr
    mutable_config.TRAIN.OPTIMIZER.MOMENTUM = momentum
    mutable_config.DATA.BATCH_SIZE = batch_size
    
    # Freeze the config again
    mutable_config.freeze()
    
    try:
        # Build model using mutable_config
        model = build_model(mutable_config)
        
        # Modify activation function based on suggestion
        for module in model.modules():
            if isinstance(module, nn.Sequential):
                for i, layer in enumerate(module):
                    if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.Tanh):
                        if activation == "relu":
                            module[i] = nn.ReLU(inplace=True)
                        elif activation == "sigmoid":
                            module[i] = nn.Sigmoid()
                        elif activation == "tanh":
                            module[i] = nn.Tanh()
        
        # Move model to GPU
        model = model.cuda()
        
        # Build optimizer and criterion
        optimizer = build_optimizer(mutable_config, model)
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = CosineAnnealingLR(
            optimizer, 
            mutable_config.TRAIN.EPOCHS, 
            eta_min=mutable_config.TRAIN.MIN_LR
        )
        
        # Rebuild data loaders with new batch size
        dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(mutable_config)
        
        # Train for a few epochs
        best_acc = 0.0
        n_epochs = tune_epochs
        
        for epoch in range(n_epochs):
            train_loss, train_acc = train_one_epoch(
                mutable_config, model, criterion, data_loader_train, optimizer, epoch
            )
            val_loss, val_acc = validate(mutable_config, data_loader_val, model)
            
            # Report intermediate metric
            trial.report(val_acc, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            best_acc = max(best_acc, val_acc)
            lr_scheduler.step()
        
        # Make sure to release CUDA memory after the trial
        torch.cuda.empty_cache()
        
        return best_acc
    except RuntimeError as e:
        # Handle CUDA OOM errors
        if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
            print(f"Trial failed due to CUDA memory error with batch size {batch_size}")
            torch.cuda.empty_cache()  # Clear CUDA cache
            return float('-inf')  # Return a very low score to avoid this configuration
        else:
            raise  # Re-raise other runtime errors


def main():
    args, config = parse_option()
    
    # Use args directly instead of adding to config
    tune_epochs = args.tune_epochs
    
    # Create study name and output directory
    os.makedirs("optuna_results", exist_ok=True)
    study_name = args.study_name
    
    # Add this line to ensure CUDA cache is empty at the start
    torch.cuda.empty_cache()
    
    # Build data loaders (will be rebuilt in each trial with different batch sizes)
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(config)
    
    # Create the study and optimize
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Add memory cleanup after each trial
    def objective_with_cleanup(trial):
        try:
            result = objective(trial, config, data_loader_train, data_loader_val, tune_epochs)
            return result
        finally:
            torch.cuda.empty_cache()
    
    study.optimize(
        objective_with_cleanup,
        n_trials=args.n_trials,
        timeout=None
    )
    
    # Print statistics
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save study results
    with open(f"optuna_results/{study_name}_results.json", "w") as f:
        json.dump({
            "best_trial": {
                "value": trial.value,
                "params": trial.params
            },
            "all_trials": [
                {
                    "trial_id": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state)
                }
                for t in study.trials
            ]
        }, f, indent=2)
    
    # Create a command line with the best parameters
    best_cmd = f"python main.py --cfg=configs/lenet_base.yaml --opts"
    best_cmd += f" DATA.BATCH_SIZE {trial.params['batch_size']}"
    best_cmd += f" TRAIN.LR {trial.params['learning_rate']}"
    best_cmd += f" TRAIN.OPTIMIZER.MOMENTUM {trial.params['momentum']}"
    best_cmd += f" MODEL.NUM_CLASSES 10 TRAIN.EPOCHS 30"
    
    print("\nRun the following command with the best hyperparameters:")
    print(best_cmd)
    
    # Optional: Create optimization history plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(f"optuna_results/{study_name}_history.png")
        
        # Plot parameter importances
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(f"optuna_results/{study_name}_importance.png")
        
        # Plot intermediate values
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_intermediate_values(study)
        plt.tight_layout()
        plt.savefig(f"optuna_results/{study_name}_intermediate.png")
        
    except (ImportError, ModuleNotFoundError):
        print("Matplotlib not available. Skipping visualization.")


if __name__ == "__main__":
    main() 