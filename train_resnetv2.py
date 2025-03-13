#!/usr/bin/env python3
import os
import time
import datetime
import argparse
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from models.resnetv2 import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    mixup_data, cutmix_data, label_smoothing_loss,
    cosine_annealing_lr, step_lr, warmup_cosine_annealing_lr, one_cycle_lr
)
from data.datasets import MediumImagenetHDF5Dataset


class Config:
    """Configuration class to store the configuration from a yaml file."""
    def __init__(self):
        pass
    
    def update(self, yaml_dict):
        for k, v in yaml_dict.items():
            if isinstance(v, dict):
                if not hasattr(self, k):
                    setattr(self, k, Config())
                getattr(self, k).update(v)
            else:
                setattr(self, k, v)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNetV2 on Medium ImageNet')
    parser.add_argument('--cfg', type=str, default='configs/resnetv2_medium_imagenet.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--output', type=str, default='',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--opts', nargs='*', default=None,
                        help='Modify config options using the command-line')
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config()
    
    # Handle base config inheritance
    if 'BASE' in config_dict:
        base_configs = config_dict.pop('BASE')
        base_configs = [base_configs] if not isinstance(base_configs, list) else base_configs
        
        for base_config in base_configs:
            base_path = os.path.join(os.path.dirname(config_file), base_config)
            with open(base_path, 'r') as f:
                config.update(yaml.safe_load(f))
    
    # Update with current config
    config.update(config_dict)
    return config


def update_config_from_opts(config, opts):
    """Update config with command line options"""
    if opts is None:
        return config
    
    # Process options in pairs (key, value)
    for i in range(0, len(opts), 2):
        if i + 1 < len(opts):
            key = opts[i]
            value = opts[i+1]
            
            # Improved type conversion handling
            try:
                # First try converting to float (handles scientific notation)
                converted = float(value)
                if '.' not in value and 'e' not in value.lower():
                    # If original value was integer-like, convert to int
                    value = int(converted) if converted.is_integer() else converted
                else:
                    value = converted
            except ValueError:
                # Handle boolean values
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    # Leave as string if not convertible
                    pass
            
            # Force float conversion for specific numerical fields
            if any(key.endswith(x) for x in ['.LR', '.MIN_LR', '.WEIGHT_DECAY', '.GAMMA', '.STEP_SIZE']):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass

            # Handle nested attributes
            keys = key.split('.')
            cfg = config
            for k in keys[:-1]:
                if not hasattr(cfg, k):
                    setattr(cfg, k, Config())
                cfg = getattr(cfg, k)
            
            # Set the value
            setattr(cfg, keys[-1], value)
    
    return config


def get_model(config):
    """Create model based on config"""
    variant = config.MODEL.VARIANT
    kwargs = {
        'num_classes': config.MODEL.NUM_CLASSES,
        'zero_init_residual': config.MODEL.ZERO_INIT_RESIDUAL,
        'use_se': config.MODEL.USE_SE,
        'dropout_rate': config.MODEL.DROP_RATE
    }
    
    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }
    
    if variant not in model_dict:
        raise ValueError(f"Unsupported model variant: {variant}")
        
    return model_dict[variant](**kwargs)


def get_optimizer(config, model):
    """
    Create optimizer based on config
    
    Args:
        config (Config): Configuration object
        model (nn.Module): Model to optimize
        
    Returns:
        optimizer (Optimizer): PyTorch optimizer
    """
    optimizer_name = config.TRAIN.OPTIMIZER.NAME
    lr = config.TRAIN.LR
    weight_decay = config.TRAIN.OPTIMIZER.WEIGHT_DECAY if hasattr(config.TRAIN.OPTIMIZER, 'WEIGHT_DECAY') else 0.01
    
    if optimizer_name == 'sgd':
        momentum = config.TRAIN.OPTIMIZER.MOMENTUM
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        betas = eval(config.TRAIN.OPTIMIZER.BETAS) if hasattr(config.TRAIN.OPTIMIZER, 'BETAS') else (0.9, 0.999)
        eps = config.TRAIN.OPTIMIZER.EPS if hasattr(config.TRAIN.OPTIMIZER, 'EPS') else 1e-8
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        betas = eval(config.TRAIN.OPTIMIZER.BETAS) if hasattr(config.TRAIN.OPTIMIZER, 'BETAS') else (0.9, 0.999)
        eps = config.TRAIN.OPTIMIZER.EPS if hasattr(config.TRAIN.OPTIMIZER, 'EPS') else 1e-8
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def get_lr_scheduler(config, optimizer):
    """Create learning rate scheduler based on config"""
    scheduler_name = config.TRAIN.LR_SCHEDULER.NAME
    
    schedulers = {
        'cosine': lambda: optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: cosine_annealing_lr(
                epoch, config.TRAIN.EPOCHS, 1.0, config.TRAIN.MIN_LR / config.TRAIN.LR
            )
        ),
        'step': lambda: optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.TRAIN.LR_SCHEDULER.STEP_SIZE, 
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA
        ),
        'warmup_cosine': lambda: optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: warmup_cosine_annealing_lr(
                epoch, config.TRAIN.EPOCHS, config.TRAIN.WARMUP_EPOCHS,
                1.0, config.TRAIN.MIN_LR / config.TRAIN.LR
            )
        ),
        'one_cycle': lambda: optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: one_cycle_lr(
                epoch, config.TRAIN.EPOCHS, 1.0, 
                config.TRAIN.LR_SCHEDULER.MAX_LR / config.TRAIN.LR,
                config.TRAIN.MIN_LR / config.TRAIN.LR
            )
        )
    }
    
    if scheduler_name not in schedulers:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
    return schedulers[scheduler_name]()


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, config, device, writer):
    """
    Train model for one epoch
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): PyTorch optimizer
        scheduler (LRScheduler): Learning rate scheduler
        epoch (int): Current epoch
        config (Config): Configuration object
        device (torch.device): Device to train on
        writer (SummaryWriter): TensorBoard writer
        
    Returns:
        train_loss (float): Training loss
        train_acc (float): Training accuracy
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    
    # Get augmentation parameters
    use_mixup = hasattr(config.AUG, 'MIXUP') and config.AUG.MIXUP > 0
    use_cutmix = hasattr(config.AUG, 'CUTMIX') and config.AUG.CUTMIX > 0
    use_label_smoothing = hasattr(config.AUG, 'LABEL_SMOOTHING') and config.AUG.LABEL_SMOOTHING > 0
    
    mixup_alpha = config.AUG.MIXUP if use_mixup else 0
    cutmix_alpha = config.AUG.CUTMIX if use_cutmix else 0
    smoothing = config.AUG.LABEL_SMOOTHING if use_label_smoothing else 0
    
    print(f"Epoch: {epoch}")
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup or cutmix randomly
        if use_mixup and use_cutmix:
            if np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
                use_mix = 'mixup'
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
                use_mix = 'cutmix'
        elif use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            use_mix = 'mixup'
        elif use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
            use_mix = 'cutmix'
        else:
            use_mix = None
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        if use_mix:
            # For mixup and cutmix, blend the losses
            if use_label_smoothing:
                loss = lam * label_smoothing_loss(outputs, targets_a, smoothing) + \
                      (1 - lam) * label_smoothing_loss(outputs, targets_b, smoothing)
            else:
                loss = lam * F.cross_entropy(outputs, targets_a) + \
                      (1 - lam) * F.cross_entropy(outputs, targets_b)
        else:
            # No mixing, just regular loss
            if use_label_smoothing:
                loss = label_smoothing_loss(outputs, targets, smoothing)
            else:
                loss = F.cross_entropy(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update stats
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        
        if use_mix:
            # For mixup, we can't easily compute accuracy during training
            # So we just use the original targets for visualization
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        else:
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if (batch_idx + 1) % config.PRINT_FREQ == 0:
            elapsed_time = time.time() - start_time
            print(f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {train_loss/(batch_idx+1):.3f} | "
                  f"Acc: {100.*correct/total:.3f}% | "
                  f"Time: {elapsed_time:.2f}s")
        
        # Update learning rate for schedulers that update per step
        if hasattr(scheduler, 'step_update'):
            scheduler.step_update(epoch * len(train_loader) + batch_idx)
    
    # Calculate average metrics
    train_loss /= (batch_idx + 1)
    train_acc = 100. * correct / total
    
    # Update TensorBoard
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/acc', train_acc, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    
    print(f"Training: Loss: {train_loss:.3f} | Acc: {train_acc:.3f}%")
    
    return train_loss, train_acc


def validate(model, val_loader, epoch, config, device, writer):
    """Validate model on validation set"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate and log metrics
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/acc', val_acc, epoch)
    
    print(f"Validation: Loss: {val_loss:.3f} | Acc: {val_acc:.3f}%")
    
    return val_loss, val_acc


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load and update config
    config = load_config(args.cfg)
    config = update_config_from_opts(config, args.opts)
    if args.output:
        config.OUTPUT = args.output
    
    # Create output directory and setup
    output_dir = Path(config.OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create TensorBoard writer and save config
    writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(config), f)
    
    # Create model, datasets, dataloaders, optimizer and scheduler
    model = get_model(config).to(device)
    print(f"Model: {config.MODEL.VARIANT}, Classes: {config.MODEL.NUM_CLASSES}")
    
    img_size = config.DATA.IMG_SIZE
    train_dataset = MediumImagenetHDF5Dataset(
        img_size=img_size, split='train', 
        filepath=config.DATA.MEDIUM_IMAGENET_PATH, augment=True
    )
    val_dataset = MediumImagenetHDF5Dataset(
        img_size=img_size, split='val',
        filepath=config.DATA.MEDIUM_IMAGENET_PATH, augment=False
    )
    
    # Create dataloaders
    batch_size = config.DATA.BATCH_SIZE
    num_workers = config.DATA.NUM_WORKERS if hasattr(config.DATA, 'NUM_WORKERS') else 4
    pin_memory = config.DATA.PIN_MEMORY if hasattr(config.DATA, 'PIN_MEMORY') else False
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # Create optimizer
    optimizer = get_optimizer(config, model)
    
    # Create learning rate scheduler
    scheduler = get_lr_scheduler(config, optimizer)
    
    # Initialize best accuracy
    best_acc = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch - 1}")
    
    # Evaluate only if specified
    if args.eval:
        print("Evaluating model...")
        validate(model, val_loader, 0, config, device, writer)
        return
    
    # Train model
    print(f"Starting training for {config.TRAIN.EPOCHS} epochs")
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch, config, device, writer
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, epoch, config, device, writer)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        save_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
        }
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0 or is_best:
            checkpoint_path = output_dir / f'checkpoint-{epoch:03d}.pth'
            torch.save(save_state, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = output_dir / 'model_best.pth'
            torch.save(save_state, best_path)
            print(f"Saved best model with accuracy {best_acc:.2f}%")
    
    # Final message
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")
    writer.close()


if __name__ == '__main__':
    main() 