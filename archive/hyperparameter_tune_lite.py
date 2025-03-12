"""
hyperparameter_tune_lite.py
~~~~~~~~~~~~~~~~~~~~~~

You can adjust these parameters:
--n-trials: Number of different hyperparameter combinations to try (default: 20)
--tune-epochs: Number of epochs to train each trial (default: 5)
--study-name: Name for your study (for saving results)

After running conda activate vision-zoo, run this script with:
python hyperparameter_tune_lite.py --cfg=config/lenet_base.yaml --opts --n-trials 10 --tune-epochs 2 --study-name "lenet_tuning_lite"

pip dependencies: optuna

Simplified hyperparameter tuning script with reduced memory usage. 
The original hyperparameter_tune.py script crashed my system. 
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
    parser = argparse.ArgumentParser("Lightweight hyperparameter tuning", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials for hyperparameter search")
    parser.add_argument("--tune-epochs", type=int, default=2, help="Number of epochs for each trial")
    parser.add_argument("--study-name", type=str, default="lenet_tuning_lite", help="Name of the study")
    
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    return args, config


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch):
    model.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    for idx, (images, target) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)):
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
        
        # Print progress more frequently
        if idx % 10 == 0:
            print(f"\rBatch {idx}/{len(data_loader)}, Loss: {loss.item():.4f}, Acc: {acc1.item():.2f}%", end="")
            
    print()  # New line after progress updates
    return losses.avg, accs.avg


def validate(config, data_loader, model):
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for idx, (images, target) in enumerate(tqdm(data_loader, desc="Validation", leave=False)):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            output = model(images)
            loss = criterion(output, target)
            
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            accs.update(acc1.item(), images.size(0))
            
    return losses.avg, accs.avg


def objective(trial, config, dataset_train, dataset_val, tune_epochs):
    # Simplified parameter space with lower memory requirements
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    # Keep batch sizes very small to avoid OOM
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Create a copy of the config and make it mutable
    mutable_config = copy.deepcopy(config)
    mutable_config.defrost()
    
    # Modify config
    mutable_config.TRAIN.LR = lr
    mutable_config.TRAIN.OPTIMIZER.MOMENTUM = momentum
    mutable_config.DATA.BATCH_SIZE = batch_size
    
    # Freeze the config
    mutable_config.freeze()
    
    print(f"\nTrial {trial.number}: LR={lr:.6f}, Momentum={momentum:.3f}, Batch Size={batch_size}")
    
    try:
        # Make sure GPU is clean before starting
        torch.cuda.empty_cache()
        
        # Create data loaders with the new batch size
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Reduced workers
            pin_memory=True
        )
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,  # Reduced workers
            pin_memory=True
        )
        
        # Build model
        model = build_model(mutable_config)
        model = model.cuda()
        
        # Build optimizer and criterion
        optimizer = build_optimizer(mutable_config, model)
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = CosineAnnealingLR(
            optimizer, 
            mutable_config.TRAIN.EPOCHS, 
            eta_min=mutable_config.TRAIN.MIN_LR
        )
        
        # Train for a few epochs
        best_acc = 0.0
        
        for epoch in range(tune_epochs):
            # Training
            train_loss, train_acc = train_one_epoch(
                mutable_config, model, criterion, data_loader_train, optimizer, epoch
            )
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            
            # Validation
            val_loss, val_acc = validate(mutable_config, data_loader_val, model)
            print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            
            best_acc = max(best_acc, val_acc)
            
            # Report to Optuna
            trial.report(val_acc, epoch)
            
            # Handle pruning
            if trial.should_prune():
                print("Trial pruned!")
                raise optuna.exceptions.TrialPruned()
            
            lr_scheduler.step()
        
        # Clean up to avoid memory leaks
        del model, optimizer, criterion, lr_scheduler, data_loader_train, data_loader_val
        torch.cuda.empty_cache()
        
        return best_acc
        
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
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    
    print(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, config, dataset_train, dataset_val, tune_epochs),
        n_trials=args.n_trials
    )
    
    # Print results
    print("\n==== Study Results ====")
    print(f"Number of finished trials: {len(study.trials)}")
    
    if study.best_trial:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value:.2f}%")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best command
        best_cmd = f"python main.py --cfg=configs/lenet_base.yaml --opts"
        best_cmd += f" DATA.BATCH_SIZE {trial.params['batch_size']}"
        best_cmd += f" TRAIN.LR {trial.params['learning_rate']}"
        best_cmd += f" TRAIN.OPTIMIZER.MOMENTUM {trial.params['momentum']}"
        best_cmd += f" MODEL.NUM_CLASSES 10 TRAIN.EPOCHS 20"
        
        print("\nRun this command with best parameters:")
        print(best_cmd)
        
        # Save results
        with open(f"optuna_results/{study_name}_results.json", "w") as f:
            json.dump({
                "best_trial": {
                    "value": trial.value,
                    "params": trial.params,
                    "command": best_cmd
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
    else:
        print("No successful trials completed")


if __name__ == "__main__":
    main() 