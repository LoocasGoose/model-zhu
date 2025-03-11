"""
hyperparameter_tune_medium.py
Simplified script with reduced memory usage. 
Original hyperparameter_tune.py script crashed my system. 
~~~~~~~~~~~~~~~~~~~~~~

You can adjust these parameters:
--n-trials: Number of different hyperparameter combinations to try (default: 20)
--tune-epochs: Number of epochs to train each trial (default: 5)
--study-name: Name for your study (for saving results)

This script is running 4 trials in parallel. The numbers are hard coded in. 
You can edit the values at `n_jobs = 4`.

After running conda activate vision-zoo, run this script with:
python hyperparameter_tune_medium.py --cfg=configs/lenet_base.yaml --n-trials 30 --tune-epochs 4 --study-name lenet_parallel_tuning

Upon finishing training, the script will print the best hyperparameters and the command to 
run to train the model with the best hyperparameters.

pip dependencies: optuna
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
    parser = argparse.ArgumentParser("Enhanced hyperparameter tuning", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials for hyperparameter search")
    parser.add_argument("--tune-epochs", type=int, default=8, help="Number of epochs for each trial")
    parser.add_argument("--study-name", type=str, default="lenet_tuning_enhanced", help="Name of the study")
    
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
        
        # Print progress less frequently for larger datasets
        if idx % 20 == 0:
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
    # Expanded parameter space with better ranges
    lr = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.85, 0.99)
    
    # Use smaller batch sizes when running in parallel
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 96, 128, 256, 512, 1024, 2048])
    activation = trial.suggest_categorical("activation", ["sigmoid"])
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-9, log=True)
    
    # Create a copy of the config and make it mutable
    mutable_config = copy.deepcopy(config)
    mutable_config.defrost()
    
    # Modify config
    mutable_config.TRAIN.LR = lr
    mutable_config.TRAIN.OPTIMIZER.MOMENTUM = momentum
    mutable_config.TRAIN.OPTIMIZER.WEIGHT_DECAY = weight_decay
    mutable_config.DATA.BATCH_SIZE = batch_size
    mutable_config.DATA.NUM_WORKERS = 2  # Reduced workers per trial when running in parallel
    
    # Freeze the config
    mutable_config.freeze()
    
    print(f"\nTrial {trial.number}: LR={lr:.6f}, Momentum={momentum:.3f}, Batch Size={batch_size}, "
          f"Weight Decay={weight_decay:.6f}, Activation={activation}")
    
    try:
        # Make sure GPU is clean before starting
        torch.cuda.empty_cache()
        
        # Create data loaders with the new batch size
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # More workers to use more CPU and memory
            pin_memory=True
        )
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,  # More workers
            pin_memory=True
        )
        
        # Build model
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
        
        model = model.cuda()
        
        # Build optimizer and criterion
        optimizer = build_optimizer(mutable_config, model)
        criterion = nn.CrossEntropyLoss()
        
        # Use more epochs in the scheduler to avoid aggressive LR reduction
        lr_scheduler = CosineAnnealingLR(
            optimizer, 
            tune_epochs * 2,  # Double the epochs for scheduler to slow down LR decay
            eta_min=mutable_config.TRAIN.MIN_LR
        )
        
        # Train for more epochs to address learning curve issues
        acc_history = []
        best_acc = 0.0
        
        # Print memory usage at start
        memory_start = torch.cuda.memory_allocated() / (1024**3)  # GB
        print(f"GPU memory at start: {memory_start:.2f} GB")
        
        for epoch in range(tune_epochs):
            # Training
            train_loss, train_acc = train_one_epoch(
                mutable_config, model, criterion, data_loader_train, optimizer, epoch
            )
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
            
            # Validation
            val_loss, val_acc = validate(mutable_config, data_loader_val, model)
            print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            acc_history.append(val_acc)
            best_acc = max(best_acc, val_acc)
            
            # Report to Optuna
            trial.report(val_acc, epoch)
            
            # Only prune after at least 4 epochs to address learning curve issues
            if trial.should_prune() and epoch >= 2:
                print("Trial pruned!")
                raise optuna.exceptions.TrialPruned()
            
            lr_scheduler.step()
            
        # Print memory peak
        memory_peak = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"Peak GPU memory usage: {memory_peak:.2f} GB")
        
        # Print learning curve
        print(f"Accuracy history: {[f'{acc:.2f}%' for acc in acc_history]}")
        
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
    
    # Define the number of parallel jobs based on GPU memory
    n_jobs = 20  # Since you're using ~1.6GB per trial on 8GB VRAM, 4 parallel jobs should be safe
    
    print(f"Starting optimization with {args.n_trials} trials, {tune_epochs} epochs each...")
    print(f"Running {n_jobs} trials in parallel!")
    
    study.optimize(
        lambda trial: objective(trial, config, dataset_train, dataset_val, tune_epochs),
        n_trials=args.n_trials,
        n_jobs=n_jobs  # Run multiple trials in parallel
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
        
        # Save best command for full training
        best_cmd = f"python main.py --cfg=configs/lenet_base.yaml --opts"
        best_cmd += f" DATA.BATCH_SIZE {trial.params['batch_size']}"
        best_cmd += f" TRAIN.LR {trial.params['learning_rate']}"
        best_cmd += f" TRAIN.OPTIMIZER.MOMENTUM {trial.params['momentum']}"
        best_cmd += f" MODEL.NUM_CLASSES 10 TRAIN.EPOCHS 20 DATA.NUM_WORKERS 12"
        
        print("\nRun this command with best parameters for full training:")
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
            
        # Try to generate visualization if matplotlib is available
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
            
            print(f"Saved visualization plots to optuna_results/{study_name}_*.png")
        except:
            print("Matplotlib visualization skipped (not available)")
    else:
        print("No successful trials completed")


if __name__ == "__main__":
    main() 