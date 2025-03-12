"""
hyperparameter_tune_alexnet.py
Script for tuning hyperparameters specifically for AlexNet.

python hyperparameter_tune_alexnet.py --cfg=configs/alexnet.yaml --n-trials 10 --tune-epochs 8 --study-name alexnet_tuning
"""

import argparse
import os
import json
import torch
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.utils.metrics import AverageMeter, accuracy
from tqdm import tqdm

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
    
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    return args, config

def objective(trial, config, dataset_train, dataset_val, tune_epochs):
    # Define hyperparameters specific to AlexNet
    lr = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
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
    
    # Assuming `best_acc` is the best accuracy achieved during training
    best_acc = train_and_validate(model, dataset_train, dataset_val, tune_epochs)  # Replace with your training logic
    
    return best_acc  # Ensure this returns a float

def train_and_validate(model, dataset_train, dataset_val, tune_epochs):
    # Move model to GPU
    model = model.cuda()  # Ensure the model is on the GPU
    
    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Example loss function
    
    best_acc = 0.0
    
    for epoch in range(tune_epochs):
        # Training loop
        model.train()
        for images, targets in data_loader_train:
            images, targets = images.cuda(), targets.cuda()  # Ensure data is on the GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in data_loader_val:
                images, targets = images.cuda(), targets.cuda()  # Ensure data is on the GPU
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        acc = 100 * correct / total
        best_acc = max(best_acc, acc)
    
    return best_acc

if __name__ == "__main__":
    args, config = parse_option()
    dataset_train, dataset_val, _, _, _, _ = build_loader(config)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config, dataset_train, dataset_val, args.tune_epochs), n_trials=args.n_trials) 