#!/usr/bin/env python
"""
DenseNet Training Script for Medium ImageNet Dataset

A specialized training script for DenseNet models with attention mechanisms
that supports all the DenseNet-specific configuration options.
"""
import os
import argparse
import datetime
import time
import json
import logging
import yaml
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from timm.utils.metrics import AverageMeter, accuracy
from tqdm import tqdm

from models.densenet import DenseNet121, DenseNet169, DenseNet201
from data.datasets import MediumImagenetHDF5Dataset


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DenseNetTraining")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train DenseNet with advanced features on medium-imagenet")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--opts", nargs="+", help="Modify config options from command line")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output/densenet", help="Output directory")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of training data to use (0.0-1.0). Use smaller values for faster experiments.")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode: 10% of data, 5 epochs, small batch size")
    return parser.parse_args()


def load_config(config_path: str, cmd_opts: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and override with command line options.
    
    Args:
        config_path: Path to the YAML config file
        cmd_opts: List of command line options in format [key1, val1, key2, val2, ...]
    
    Returns:
        Dictionary containing configuration
    """
    # Load base config from YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line options if provided
    if cmd_opts and len(cmd_opts) > 0:
        i = 0
        while i < len(cmd_opts):
            key = cmd_opts[i]
            if i + 1 < len(cmd_opts):
                val = cmd_opts[i + 1]
                # Handle key with dots (nested config)
                if '.' in key:
                    parts = key.split('.')
                    cur = config
                    for part in parts[:-1]:
                        if part not in cur:
                            cur[part] = {}
                        cur = cur[part]
                    
                    # Try to parse the value to the right type
                    try:
                        # Try as numeric
                        if val.lower() == 'true':
                            val = True
                        elif val.lower() == 'false':
                            val = False
                        elif '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except (ValueError, AttributeError):
                        # Keep as string if not numeric
                        pass
                    
                    cur[parts[-1]] = val
                else:
                    # Same for top-level keys
                    try:
                        if val.lower() == 'true':
                            val = True
                        elif val.lower() == 'false':
                            val = False
                        elif '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except (ValueError, AttributeError):
                        pass
                    
                    config[key] = val
            i += 2
    
    return config


def build_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Build DenseNet model based on configuration.
    
    Args:
        config: Dictionary containing model configuration
        device: Device to put the model on
    
    Returns:
        DenseNet model
    """
    # Get model type/variation
    model_type = config.get('VARIATION', config.get('MODEL', {}).get('TYPE', '121'))
    
    # Get model parameters
    model_config = config.get('MODEL', {})
    num_classes = model_config.get('NUM_CLASSES', 1000)
    attention = model_config.get('ATTENTION', 'se')
    activation = model_config.get('ACTIVATION', 'swish')
    attention_pooling = model_config.get('ATTENTION_POOLING', False)
    stochastic_depth = model_config.get('STOCHASTIC_DEPTH', 0.0)
    dropout_rate = model_config.get('DROP_RATE', 0.2)
    small_inputs = model_config.get('SMALL_INPUTS', True)
    
    # Create model based on type
    logger.info(f"Creating DenseNet-{model_type} model")
    if model_type == '169':
        model = DenseNet169(
            num_classes=num_classes,
            small_inputs=small_inputs,
            use_attention=attention,
            activation=activation,
            use_attention_pooling=attention_pooling,
            stochastic_depth_prob=stochastic_depth,
            dropout_rate=dropout_rate
        )
    elif model_type == '201':
        model = DenseNet201(
            num_classes=num_classes,
            small_inputs=small_inputs,
            use_attention=attention,
            activation=activation,
            use_attention_pooling=attention_pooling,
            stochastic_depth_prob=stochastic_depth,
            dropout_rate=dropout_rate
        )
    else:  # Default to 121
        model = DenseNet121(
            num_classes=num_classes,
            small_inputs=small_inputs,
            use_attention=attention,
            activation=activation,
            use_attention_pooling=attention_pooling,
            stochastic_depth_prob=stochastic_depth,
            dropout_rate=dropout_rate
        )
    
    # Log model configuration
    logger.info(f"Model parameters: "
                f"attention={attention}, "
                f"activation={activation}, "
                f"attention_pooling={attention_pooling}, "
                f"stochastic_depth={stochastic_depth}, "
                f"dropout_rate={dropout_rate}")
    
    model = model.to(device)
    
    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {n_parameters/1e6:.2f}M")
    
    return model


def build_dataloaders(config: Dict[str, Any], subset_ratio: float = 1.0) -> Tuple:
    """
    Build data loaders for training and validation.
    
    Args:
        config: Dictionary containing dataset configuration
        subset_ratio: Fraction of training data to use (0.0-1.0)
    
    Returns:
        Tuple of train and validation data loaders
    """
    data_config = config.get('DATA', {})
    
    # Get dataset path
    dataset_path = data_config.get('MEDIUM_IMAGENET_PATH', '/honey/nmep/medium-imagenet-96.hdf5')
    img_size = data_config.get('IMG_SIZE', 96)
    batch_size = data_config.get('BATCH_SIZE', 64)
    num_workers = data_config.get('NUM_WORKERS', 4)
    pin_memory = data_config.get('PIN_MEMORY', True)
    
    logger.info(f"Loading dataset from {dataset_path} with image size {img_size}")
    
    # Create datasets for train, val, and test splits
    dataset_train = MediumImagenetHDF5Dataset(
        img_size=img_size,
        split="train",
        filepath=dataset_path,
        augment=True
    )
    
    dataset_val = MediumImagenetHDF5Dataset(
        img_size=img_size,
        split="val",
        filepath=dataset_path,
        augment=False
    )
    
    dataset_test = MediumImagenetHDF5Dataset(
        img_size=img_size,
        split="test",
        filepath=dataset_path,
        augment=False
    )
    
    # Create subset for faster training if requested
    if subset_ratio < 1.0:
        # Calculate subset size
        train_size = int(len(dataset_train) * subset_ratio)
        val_size = int(len(dataset_val) * subset_ratio)
        
        # Create random indices as lists
        train_indices = torch.randperm(len(dataset_train))[:train_size].tolist()
        val_indices = torch.randperm(len(dataset_val))[:val_size].tolist()
        
        # Create subset datasets
        dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
        dataset_val = torch.utils.data.Subset(dataset_val, val_indices)
        
        logger.info(f"Using {subset_ratio:.1%} of data: {train_size} training samples, {val_size} validation samples")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    logger.info(f"Train dataset size: {len(dataset_train)}, batch size: {batch_size}, workers: {num_workers}")
    logger.info(f"Validation dataset size: {len(dataset_val)}")
    logger.info(f"Test dataset size: {len(dataset_test)}")
    
    return dataset_train, dataset_val, dataset_test, train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    print_freq: int = 100
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP
        print_freq: Frequency of printing batch results
    
    Returns:
        Tuple of (accuracy, loss) for the epoch
    """
    model.train()
    
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    batch_time = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for idx, (samples, targets) in pbar:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        if use_amp:
            # Mixed precision forward pass
            with autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
            
            # Scale loss and backward
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            # Standard forward pass
            outputs = model(samples)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
        
        # Measure accuracy and record loss
        acc1, = accuracy(outputs, targets)
        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            pbar.set_description(
                f"Epoch: [{epoch}/{total_epochs}][{idx}/{len(dataloader)}] "
                f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"Acc@1: {acc1_meter.val:.3f} ({acc1_meter.avg:.3f}) "
                f"LR: {lr:.6f} "
                f"Time: {batch_time.val:.4f}s "
                f"Mem: {memory_used:.0f}MB"
            )
    
    epoch_time = time.time() - start
    logger.info(f"Epoch: [{epoch}/{total_epochs}] completed in {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f"Train Acc@1: {acc1_meter.avg:.3f} Loss: {loss_meter.avg:.4f}")
    
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model on the validation set.
    
    Args:
        model: Model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Tuple of (accuracy, loss) for the validation set
    """
    model.eval()
    
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for idx, (samples, targets) in pbar:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(samples)
        loss = criterion(outputs, targets)
        
        # Measure accuracy and record loss
        acc1, = accuracy(outputs, targets)
        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        pbar.set_description(
            f"Val: [{idx}/{len(dataloader)}] "
            f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
            f"Acc@1: {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})"
        )
    
    logger.info(f"Validation Acc@1: {acc1_meter.avg:.3f} Loss: {loss_meter.avg:.4f}")
    
    return acc1_meter.avg, loss_meter.avg


def save_checkpoint(
    output_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    acc: float,
    loss: float,
    model_type: str,
    is_best: bool = False
) -> None:
    """
    Save a checkpoint of the model.
    
    Args:
        output_dir: Directory to save checkpoint
        epoch: Current epoch
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        acc: Validation accuracy
        loss: Validation loss
        model_type: Model type (e.g., '121')
        is_best: Whether this is the best model so far
    """
    os.makedirs(output_dir, exist_ok=True)
    
    save_state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'acc': acc,
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(output_dir, f'densenet_{model_type}_epoch_{epoch}.pth')
    torch.save(save_state, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(output_dir, f'densenet_{model_type}_best.pth')
        torch.save(save_state, best_path)
        logger.info(f"Best model saved: {best_path}")


def main():
    """Main function to train and evaluate the model"""
    args = parse_args()
    config = load_config(args.cfg, args.opts)
    
    # Enable fast mode if requested
    subset_ratio = args.subset
    if args.fast:
        # Override settings for fast experimentation
        subset_ratio = 0.1  # Use 10% of the data
        config.setdefault('TRAIN', {})['EPOCHS'] = 5  # Only 5 epochs
        config.setdefault('DATA', {})['BATCH_SIZE'] = min(config.get('DATA', {}).get('BATCH_SIZE', 64), 16)  # Smaller batch size
        config.setdefault('DATA', {})['NUM_WORKERS'] = min(config.get('DATA', {}).get('NUM_WORKERS', 4), 2)  # Fewer workers
        if 'MODEL' in config:
            config['MODEL']['TYPE'] = '121'  # Use smallest model
            config['MODEL']['ATTENTION'] = 'none'  # No attention mechanisms
        config.setdefault('SAVE_FREQ', 1)  # Save each epoch
        
        logger.info("FAST MODE ENABLED: Using 10% of data, 5 epochs, small batch size, and simplified model")
    
    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model
    model = build_model(config, device)
    
    # Build dataloaders
    dataset_train, dataset_val, dataset_test, train_loader, val_loader, test_loader = build_dataloaders(config, subset_ratio)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer
    train_config = config.get('TRAIN', {})
    learning_rate = train_config.get('LR', 0.001)
    weight_decay = train_config.get('WEIGHT_DECAY', 1e-4)
    
    # Select optimizer based on config
    optimizer_name = train_config.get('OPTIMIZER', {}).get('NAME', 'adamw').lower()
    if optimizer_name == 'sgd':
        momentum = train_config.get('OPTIMIZER', {}).get('MOMENTUM', 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        logger.info(f"Using SGD optimizer with lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}")
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"Using Adam optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    else:  # Default to AdamW
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    
    # Set up scheduler
    total_epochs = train_config.get('EPOCHS', 100)
    scheduler_type = train_config.get('SCHEDULER', 'cosine')
    min_lr = train_config.get('MIN_LR', 1e-6)
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr
        )
        logger.info(f"Using CosineAnnealingLR scheduler with T_max={total_epochs}, eta_min={min_lr}")
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=min_lr
        )
        logger.info(f"Using ReduceLROnPlateau scheduler with patience=5, min_lr={min_lr}")
    else:
        scheduler = None
        logger.info("No scheduler being used")
    
    # Set up AMP (Automatic Mixed Precision)
    amp_enabled = train_config.get('AMP_OPT_LEVEL', '') != ''
    scaler = GradScaler() if amp_enabled else None
    logger.info(f"Automatic Mixed Precision (AMP): {'Enabled' if amp_enabled else 'Disabled'}")
    
    # Set up other training parameters
    print_freq = config.get('PRINT_FREQ', 100)
    save_freq = config.get('SAVE_FREQ', 5)
    model_type = config.get('VARIATION', config.get('MODEL', {}).get('TYPE', '121'))
    
    # Training loop
    logger.info(f"Starting training for {total_epochs} epochs")
    best_acc = 0.0
    
    # Create metrics.json for logging
    metrics_file = os.path.join(output_dir, 'metrics.json')
    # Clear metrics file if it exists
    if os.path.exists(metrics_file):
        open(metrics_file, 'w').close()
    
    start_time = time.time()
    for epoch in range(total_epochs):
        # Train for one epoch
        train_acc, train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=total_epochs,
            use_amp=amp_enabled,
            scaler=scaler,
            print_freq=print_freq
        )
        
        # Validate
        val_acc, val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Check if this is the best model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or epoch == total_epochs - 1 or is_best:
            save_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                acc=val_acc,
                loss=val_loss,
                model_type=model_type,
                is_best=is_best
            )
        
        # Log metrics
        metrics = {
            "epoch": epoch,
            "train_acc": float(train_acc),
            "train_loss": float(train_loss),
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]['lr']),
            "best_acc": float(best_acc)
        }
        
        # Append to metrics file
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        logger.info(f"Epoch {epoch+1}/{total_epochs} - "
                   f"Train Acc: {train_acc:.3f}, Train Loss: {train_loss:.4f}, "
                   f"Val Acc: {val_acc:.3f}, Val Loss: {val_loss:.4f}, "
                   f"Best Acc: {best_acc:.3f}, "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")
    logger.info(f"Best validation accuracy: {best_acc:.3f}")
    
    # Save the final model
    final_path = os.path.join(output_dir, f'densenet_{model_type}_final.pth')
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'acc': best_acc
    }, final_path)
    logger.info(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main() 