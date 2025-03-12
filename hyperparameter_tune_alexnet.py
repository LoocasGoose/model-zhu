"""
hyperparameter_tune_alexnet.py
Script for tuning hyperparameters specifically for AlexNet.

CUDA_VISIBLE_DEVICES=5 python hyperparameter_tune_alexnet.py --cfg=configs/alexnet.yaml --n-trials 10 --tune-epochs 8 --n-jobs 4 --subset-ratio 0.1 --early-stop-patience 2 --study-name alexnet_tuning 
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
    print(f"Trial {trial.number}: Learning Rate: {lr}, Batch Size: {batch_size}, Dropout Rate: {dropout_rate}")
    
    try:
        # Create a mutable copy of the config
        mutable_config = config.clone()
        mutable_config.defrost()
        
        # Modify config for tuning
        mutable_config.TRAIN.LR = lr
        mutable_config.DATA.BATCH_SIZE = batch_size
        mutable_config.MODEL.DROP_RATE = dropout_rate
        mutable_config.freeze()
        
        # Build model and proceed with training and validation...
        model = build_model(mutable_config)
        
        # Subset the data for faster tuning
        train_size = int(len(dataset_train) * subset_ratio)
        indices = torch.randperm(len(dataset_train))[:train_size]
        subset_train = torch.utils.data.Subset(dataset_train, indices.tolist())
        
        # Subset validation data as well
        val_size = min(len(dataset_val), 2000)  # Limit validation set to 2000 samples
        val_indices = torch.randperm(len(dataset_val))[:val_size]
        subset_val = torch.utils.data.Subset(dataset_val, val_indices.tolist())
        
        # Train and validate the model
        best_acc = train_and_validate(model, subset_train, subset_val, tune_epochs, batch_size, patience)
        
        # Log the best accuracy for the current trial
        print(f"Trial {trial.number} completed with best accuracy: {best_acc:.2f}%")
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache()
        
        return best_acc
    
    except Exception as e:
        print(f"Error in trial {trial.number}: {str(e)}")
        torch.cuda.empty_cache()
        return float('-inf')  # Return worst possible score

def train_and_validate(model, dataset_train, dataset_val, tune_epochs, batch_size, patience=2):
    print("Starting training and validation...")  # Debugging print
    model = model.cuda()  # Ensure the model is on the GPU
    
    # Create data loaders with optimized parameters
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your CPU cores
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Example loss function
    
    for epoch in range(tune_epochs):
        print(f"Epoch {epoch + 1}/{tune_epochs}")  # Log current epoch
        model.train()
        
        # Add progress bar for training
        with tqdm(total=len(data_loader_train), desc="Training", unit="batch") as pbar:
            for images, targets in data_loader_train:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # Faster than setting to zero
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                pbar.update(1)  # Update progress bar
                pbar.set_postfix(loss=loss.item())  # Display current loss in the progress bar
        
        # Validation loop with progress bar
        model.eval()
        correct = 0
        total = 0
        
        with tqdm(total=len(data_loader_val), desc="Validation", unit="batch") as pbar:
            with torch.no_grad():
                for images, targets in data_loader_val:
                    images = images.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)
                    
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    pbar.update(1)  # Update progress bar
            
        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")  # Log validation accuracy for the epoch
        
        # Check for improvement
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_acc

def main():
    args, config = parse_option()
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
    
    print(f"Starting optimization with {args.n_trials} trials, {args.tune_epochs} epochs each...")
    print(f"Using {args.subset_ratio*100:.1f}% of training data for faster tuning")
    print(f"Running {args.n_jobs} trials in parallel!")
    
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
        best_cmd = f"python main.py --cfg=configs/alexnet.yaml --opts"
        best_cmd += f" DATA.BATCH_SIZE {trial.params['batch_size']}"
        best_cmd += f" TRAIN.LR {trial.params['learning_rate']}"
        best_cmd += f" MODEL.DROP_RATE {trial.params['dropout_rate']}"
        best_cmd += f" TRAIN.EPOCHS 20"
        
        print("\nRun this command with best parameters for full training:")
        print(best_cmd)
        
        # Save results
        with open(f"optuna_results/{args.study_name}_results.json", "w") as f:
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