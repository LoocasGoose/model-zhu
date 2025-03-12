"""
hyperparameter_tune_alexnet.py
Script for tuning hyperparameters specifically for AlexNet.

CUDA_VISIBLE_DEVICES=5 python hyperparameter_tune_alexnet.py --cfg=configs/alexnet.yaml --n-trials 10 --tune-epochs 8 --study-name alexnet_tuning --n-jobs 4
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
    
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    return args, config

def objective(trial, config, dataset_train, dataset_val, tune_epochs):
    # Define hyperparameters specific to AlexNet
    lr = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    # Log the parameters for the current trial
    print(f"Trial {trial.number}: Learning Rate: {lr}, Batch Size: {batch_size}, Dropout Rate: {dropout_rate}")
    
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
    
    # Train and validate the model
    best_acc = train_and_validate(model, dataset_train, dataset_val, tune_epochs)
    
    # Log the best accuracy for the current trial
    print(f"Trial {trial.number} completed with best accuracy: {best_acc:.2f}%")
    
    return best_acc  # Ensure this returns a float

def train_and_validate(model, dataset_train, dataset_val, tune_epochs):
    print("Starting training and validation...")  # Debugging print
    model = model.cuda()  # Ensure the model is on the GPU
    
    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Example loss function
    
    best_acc = 0.0
    
    for epoch in range(tune_epochs):
        print(f"Epoch {epoch + 1}/{tune_epochs}")  # Log current epoch
        model.train()
        
        # Add progress bar for training
        with tqdm(total=len(data_loader_train), desc="Training", unit="batch") as pbar:
            for images, targets in data_loader_train:
                images, targets = images.cuda(), targets.cuda()  # Ensure data is on the GPU
                optimizer.zero_grad()
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
                    images, targets = images.cuda(), targets.cuda()  # Ensure data is on the GPU
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    pbar.update(1)  # Update progress bar
            
        acc = 100 * correct / total
        best_acc = max(best_acc, acc)
        print(f"Validation Accuracy: {acc:.2f}%")  # Log validation accuracy for the epoch
    
    return best_acc

def main():
    args, config = parse_option()
    dataset_train, dataset_val, _, _, _, _ = build_loader(config)
    
    # Create Optuna study with better pruning
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=args.tune_epochs, reduction_factor=3
        )
    )
    
    print(f"Starting optimization with {args.n_trials} trials, {args.tune_epochs} epochs each...")
    print(f"Running {args.n_jobs} trials in parallel!")
    
    study.optimize(
        lambda trial: objective(trial, config, dataset_train, dataset_val, args.tune_epochs),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs  # Run multiple trials in parallel
    )
    
    # ... existing result logging ...

if __name__ == "__main__":
    main() 