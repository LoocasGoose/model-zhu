#!/usr/bin/env python3
"""
Memory-optimized unified training script for ResNetV2 variants.
Supports resnet18v2, resnet50v2, and resnet101v2 with a clean, efficient implementation.
"""
import os
import argparse
import datetime
import time
import yaml
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging

from models.resnetv2 import (
    resnet18, resnet50, resnet101,
    mixup_data, cutmix_data, label_smoothing_loss
)
from data.datasets import MediumImagenetHDF5Dataset


class Config:
    """Configuration class to store settings from YAML files."""
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
    parser = argparse.ArgumentParser(description='Train ResNetV2 variants with memory optimization')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file (resnet18v2.yaml, resnet50v2.yaml, or resnet101v2.yaml)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--model-variant', type=str, default=None, choices=['resnet18v2', 'resnet50v2', 'resnet101v2'],
                        help='Model variant to use (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--opts', nargs='*', default=None,
                        help='Modify config options using KEY VALUE pairs')
    return parser.parse_args()


def load_config(config_file):
    """Load configuration from YAML file with inheritance support."""
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


def update_config_from_args_and_opts(config, args):
    """Update config with command line arguments and options."""
    # Override with direct arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.lr:
        config.TRAIN.LR = args.lr
    if args.epochs:
        config.TRAIN.EPOCHS = args.epochs
    if args.output:
        config.OUTPUT = args.output
    if args.workers:
        config.DATA.NUM_WORKERS = args.workers
    if args.model_variant:
        # Map user-friendly names to internal names
        variant_map = {
            'resnet18v2': 'resnet18',
            'resnet50v2': 'resnet50',
            'resnet101v2': 'resnet101'
        }
        config.MODEL.VARIANT = variant_map.get(args.model_variant, config.MODEL.VARIANT)
    
    # Override with --opts arguments
    if args.opts:
        for i in range(0, len(args.opts), 2):
            if i + 1 < len(args.opts):
                key, value = args.opts[i], args.opts[i+1]
                
                # Try to convert value to appropriate type
                try:
                    # Convert to appropriate numeric type
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except (ValueError, AttributeError):
                    # Handle boolean values
                    if isinstance(value, str) and value.lower() == 'true':
                        value = True
                    elif isinstance(value, str) and value.lower() == 'false':
                        value = False
                
                # Set nested config attribute
                keys = key.split('.')
                cfg = config
                for k in keys[:-1]:
                    if not hasattr(cfg, k):
                        setattr(cfg, k, Config())
                    cfg = getattr(cfg, k)
                setattr(cfg, keys[-1], value)
    
    return config


def setup_logger(output_dir, name="train"):
    """Set up logger for training."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, f"{name}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_model(config):
    """Build ResNetV2 model based on config."""
    variant = config.MODEL.VARIANT
    num_classes = config.MODEL.NUM_CLASSES
    use_se = getattr(config.MODEL, 'USE_SE', False)
    drop_rate = getattr(config.MODEL, 'DROP_RATE', 0.0)
    zero_init_residual = getattr(config.MODEL, 'ZERO_INIT_RESIDUAL', True)
    
    # Map standard variant names to functions
    model_dict = {
        'resnet18': resnet18,
        'resnet50': resnet50,
        'resnet101': resnet101
    }
    
    if variant not in model_dict:
        raise ValueError(f"Unsupported model variant: {variant}")
    
    # Create model
    model = model_dict[variant](
        num_classes=num_classes,
        use_se=use_se,
        dropout_rate=drop_rate,
        zero_init_residual=zero_init_residual
    )
    
    return model


def get_optimizer(config, model):
    """Create optimizer based on config."""
    optimizer_name = config.TRAIN.OPTIMIZER.NAME.lower()
    lr = config.TRAIN.LR
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=getattr(config.TRAIN.OPTIMIZER, 'NESTEROV', False)
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(config, optimizer):
    """Create learning rate scheduler based on config."""
    scheduler_name = config.TRAIN.LR_SCHEDULER.NAME.lower()
    epochs = config.TRAIN.EPOCHS
    min_lr = getattr(config.TRAIN, 'MIN_LR', 0)
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
    elif scheduler_name == 'step':
        step_size = getattr(config.TRAIN.LR_SCHEDULER, 'STEP_SIZE', epochs // 3)
        gamma = getattr(config.TRAIN.LR_SCHEDULER, 'GAMMA', 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_name == 'warmup_cosine':
        warmup_epochs = getattr(config.TRAIN, 'WARMUP_EPOCHS', epochs // 10)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return epoch / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return min_lr / config.TRAIN.LR + 0.5 * (1 - min_lr / config.TRAIN.LR) * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Default to cosine scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
    
    return scheduler


def train_one_epoch(model, train_loader, optimizer, epoch, config, device, logger, scaler=None):
    """Train the model for one epoch with memory optimizations."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Get training parameters
    use_amp = getattr(config.TRAIN, 'USE_AMP', True) and scaler is not None
    grad_accum_steps = getattr(config.TRAIN, 'GRADIENT_ACCUMULATION_STEPS', 1)
    grad_clip_val = getattr(config.TRAIN, 'GRADIENT_CLIP_VAL', 0.0)
    
    # Get augmentation parameters
    use_mixup = getattr(config.AUG, 'MIXUP', 0) > 0
    use_cutmix = getattr(config.AUG, 'CUTMIX', 0) > 0
    mixup_alpha = getattr(config.AUG, 'MIXUP', 0.2)
    cutmix_alpha = getattr(config.AUG, 'CUTMIX', 0.2)
    label_smoothing = getattr(config.AUG, 'LABEL_SMOOTHING', 0.1)
    
    # Memory optimization settings
    empty_cache_freq = getattr(config.MEMORY, 'EMPTY_CACHE_FREQ', 50) if hasattr(config, 'MEMORY') else 50
    
    # DISABLE gradient checkpointing - it can hurt training dynamics
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            if hasattr(block, 'use_checkpoint'):
                block.use_checkpoint = False
    
    # Create progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.TRAIN.EPOCHS-1}")
    
    # Set gradient accumulation counters
    optimizer.zero_grad()
    accumulation_steps = 0
    
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # Apply data augmentation (with reduced probability to improve stability)
        rand_val = torch.rand(1).item()
        use_mix = None
        
        if use_mixup and rand_val < 0.2:  # Reduced from 0.3 to 0.2
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            use_mix = 'mixup'
        elif use_cutmix and rand_val < 0.3:  # Reduced from 0.5 to 0.3
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
            use_mix = 'cutmix'
        
        # Use mixed precision for forward pass if available
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(inputs)
                
                if use_mix:
                    loss = lam * label_smoothing_loss(outputs, targets_a, label_smoothing) + \
                        (1 - lam) * label_smoothing_loss(outputs, targets_b, label_smoothing)
                else:
                    loss = label_smoothing_loss(outputs, targets, label_smoothing)
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
                
            # Backward pass with scaler
            scaler.scale(loss).backward()
            accumulation_steps += 1
            
            # Step optimizer on accumulation boundary or at the end
            if accumulation_steps == grad_accum_steps or i == len(train_loader) - 1:
                if grad_clip_val > 0:
                    # Unscale before clipping gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accumulation_steps = 0
        else:
            # Standard precision
            outputs = model(inputs)
            
            if use_mix:
                loss = lam * label_smoothing_loss(outputs, targets_a, label_smoothing) + \
                    (1 - lam) * label_smoothing_loss(outputs, targets_b, label_smoothing)
            else:
                loss = label_smoothing_loss(outputs, targets, label_smoothing)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            accumulation_steps += 1
            
            # Step optimizer on accumulation boundary or at the end
            if accumulation_steps == grad_accum_steps or i == len(train_loader) - 1:
                if grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
                optimizer.step()
                optimizer.zero_grad()
                accumulation_steps = 0
        
        # Record stats (use full loss, not scaled)
        batch_loss = loss.item() * grad_accum_steps
        total_loss += batch_loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        accuracy = 100.0 * correct / total
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            'loss': f'{total_loss/(i+1):.3f}',
            'acc': f'{accuracy:.2f}%',
            'lr': f'{current_lr:.6f}'
        })
        
        # Explicit memory cleanup
        if i % empty_cache_freq == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, device, logger):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(val_loader, desc="Validating")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Record stats
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        accuracy = 100.0 * correct / total
        pbar.set_postfix({
            'loss': f'{total_loss/len(val_loader):.3f}',
            'acc': f'{accuracy:.2f}%'
        })
    
    # Final metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    logger.info(f"Validation - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, config, output_dir, is_best=False, logger=None):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get friendly model name for filename
    variant_to_name = {
        'resnet18': 'resnet18v2',
        'resnet50': 'resnet50v2',
        'resnet101': 'resnet101v2'
    }
    model_name = variant_to_name.get(config.MODEL.VARIANT, 'resnetv2')
    
    # Save to the specific epoch checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'accuracy': accuracy,
        'config': vars(config) if hasattr(config, '__dict__') else config,
    }, checkpoint_path)
    
    # If best model so far, create a copy
    if is_best:
        best_path = os.path.join(output_dir, f"{model_name}_best.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'accuracy': accuracy,
            'config': vars(config) if hasattr(config, '__dict__') else config,
        }, best_path)
        if logger:
            logger.info(f"Saved best model with accuracy {accuracy:.2f}% to {best_path}")
    
    return checkpoint_path


def main():
    """Main function to run training."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    # Load config and update with args
    config = load_config(args.cfg)
    config = update_config_from_args_and_opts(config, args)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = config.OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(output_dir)
    
    # Log config and command
    logger.info(f"Using config: {args.cfg}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Using device: {device}")
    
    # Save config to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(config) if hasattr(config, '__dict__') else config, f)
    
    # Get human-readable model name
    variant_to_name = {
        'resnet18': 'ResNet18v2',
        'resnet50': 'ResNet50v2',
        'resnet101': 'ResNet101v2'
    }
    model_name = variant_to_name.get(config.MODEL.VARIANT, 'ResNetV2')
    
    # Create model
    model = get_model(config)
    model = model.to(device)
    logger.info(f"Created {model_name} model")
    
    # Log parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} trainable parameters")
    
    # Create datasets and dataloaders
    img_size = config.DATA.IMG_SIZE
    batch_size = config.DATA.BATCH_SIZE
    num_workers = getattr(config.DATA, 'NUM_WORKERS', 4)
    pin_memory = getattr(config.DATA, 'PIN_MEMORY', False)
    persistent_workers = getattr(config.DATA, 'PERSISTENT_WORKERS', False)
    prefetch_factor = getattr(config.DATA, 'PREFETCH_FACTOR', 2)
    
    # Dataset path
    dataset_path = config.DATA.MEDIUM_IMAGENET_PATH
    
    train_dataset = MediumImagenetHDF5Dataset(
        img_size=img_size, 
        split='train',
        filepath=dataset_path, 
        augment=True
    )
    val_dataset = MediumImagenetHDF5Dataset(
        img_size=img_size, 
        split='val',
        filepath=dataset_path, 
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # Create gradient scaler for mixed precision
    use_amp = getattr(config.TRAIN, 'USE_AMP', True) and not args.no_amp
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Get validation frequency
    validate_freq = getattr(config, 'VALIDATE_FREQ', 1)
    save_freq = getattr(config, 'SAVE_FREQ', 5)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_accuracy = 0
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict') and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_accuracy = checkpoint.get('accuracy', 0)
            logger.info(f"Resumed from epoch {start_epoch-1} with accuracy {best_accuracy:.2f}%")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting from scratch")
    
    # Evaluate only if specified
    if args.eval:
        logger.info("Evaluating model...")
        val_loss, val_acc = validate(model, val_loader, device, logger)
        logger.info(f"Evaluation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        return
    
    # Print training info
    logger.info(f"Starting training for {config.TRAIN.EPOCHS} epochs")
    logger.info(f"Batch size: {batch_size}, Accumulation steps: {getattr(config.TRAIN, 'GRADIENT_ACCUMULATION_STEPS', 1)}")
    logger.info(f"Effective batch size: {batch_size * getattr(config.TRAIN, 'GRADIENT_ACCUMULATION_STEPS', 1)}")
    logger.info(f"Mixed precision: {use_amp}, Learning rate: {config.TRAIN.LR}")
    
    # Initialize metrics tracking
    metrics_history = []
    
    # Training loop
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        # Train one epoch
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, epoch, config, device, logger, scaler
        )
        epoch_time = time.time() - epoch_start_time
        
        # Validate if needed
        validate_this_epoch = (epoch % validate_freq == 0 or epoch == config.TRAIN.EPOCHS - 1)
        if validate_this_epoch:
            val_loss, val_acc = validate(model, val_loader, device, logger)
            # Update best accuracy
            is_best = val_acc > best_accuracy
            best_accuracy = max(best_accuracy, val_acc)
        else:
            val_loss, val_acc = -1, -1
            is_best = False
            logger.info(f"Skipping validation for epoch {epoch}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint if needed
        if epoch % save_freq == 0 or epoch == config.TRAIN.EPOCHS - 1 or is_best:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, 
                config, output_dir, is_best, logger
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Track metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss if validate_this_epoch else None,
            "val_acc": val_acc if validate_this_epoch else None,
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time
        }
        metrics_history.append(metrics)
        
        # Save metrics to file
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)
        
        # Update best accuracy message
        logger.info(f"Best accuracy so far: {best_accuracy:.2f}%")
        logger.info(f"Epoch {epoch} completed in {datetime.timedelta(seconds=int(epoch_time))}")
    
    # Final message
    logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main() 