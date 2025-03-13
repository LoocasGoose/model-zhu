#!/usr/bin/env python3
"""
Unified ResNet training script.
Supports both original ResNet models from resnet.py and optimized models from resnetv2.py
with all performance optimizations for efficient training.
"""
import os
import argparse
import datetime
import json
import time
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from timm.utils.metrics import AverageMeter, accuracy
from tqdm import tqdm
import logging

# Import models
from models.resnet import ResNet18
from models.resnetv2 import resnet50, resnet18
from data import build_loader
from config import get_config
from utils import create_logger, load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser("Unified ResNet training script")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    
    # Model selection
    parser.add_argument("--model-type", type=str, default="resnetv2", 
                        choices=["resnet", "resnetv2"], 
                        help="Model architecture type")
    parser.add_argument("--model-variant", type=str, default="resnet50", 
                        choices=["resnet18", "resnet50"], 
                        help="Model size/variant")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--workers", type=int, default=16, help="Number of data loading workers")
    
    # Optimization options
    parser.add_argument("--validate-freq", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing")
    
    # Other options
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--eval", action="store_true", help="Evaluate only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--opts", nargs="+", help="Modify config options using KEY VALUE pairs")
    
    return parser.parse_args()


def build_model(args, config):
    """Build the appropriate model based on args and config"""
    num_classes = config.MODEL.NUM_CLASSES if hasattr(config.MODEL, 'NUM_CLASSES') else 200
    
    if args.model_type == "resnet":
        # Original ResNet from resnet.py
        if args.model_variant == "resnet18":
            model = ResNet18(
                num_classes=num_classes,
                enable_checkpoint=not args.no_checkpoint
            )
        else:
            raise ValueError(f"Model variant {args.model_variant} not supported for original ResNet")
    else:
        # ResNetV2 from resnetv2.py
        if args.model_variant == "resnet18":
            model = resnet18(
                num_classes=num_classes,
                zero_init_residual=getattr(config.MODEL, 'ZERO_INIT_RESIDUAL', True),
                use_se=getattr(config.MODEL, 'USE_SE', False),
                dropout_rate=getattr(config.MODEL, 'DROP_RATE', 0.0)
            )
        elif args.model_variant == "resnet50":
            model = resnet50(
                num_classes=num_classes,
                zero_init_residual=getattr(config.MODEL, 'ZERO_INIT_RESIDUAL', True),
                use_se=getattr(config.MODEL, 'USE_SE', False),
                dropout_rate=getattr(config.MODEL, 'DROP_RATE', 0.0)
            )
        else:
            raise ValueError(f"Unknown model variant: {args.model_variant}")
        
        # Enable gradient checkpointing for ResNetV2 if requested
        if not args.no_checkpoint:
            for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
                for block in layer:
                    setattr(block, 'use_checkpoint', True)
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, config, logger, scaler=None):
    """Optimized training function for one epoch"""
    model.train()
    
    # Setup metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    
    # Get training parameters
    use_amp = scaler is not None
    grad_accum_steps = config.grad_accum_steps
    grad_clip_val = getattr(config, 'grad_clip_val', 0.0)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Create progress bar
    num_steps = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs-1}", leave=True)
    
    start_time = time.time()
    end_time = time.time()
    
    # Track accumulated gradients
    accum_step = 0
    
    for idx, (inputs, targets) in enumerate(pbar):
        # Measure data loading time
        data_time.update(time.time() - end_time)
        
        # Move data to device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass with mixed precision if enabled
        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients
            accum_step += 1
            if accum_step == grad_accum_steps or idx == num_steps - 1:
                # Unscale gradients for clipping
                if grad_clip_val > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_step = 0
        else:
            # Standard precision training
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            accum_step += 1
            if accum_step == grad_accum_steps or idx == num_steps - 1:
                if grad_clip_val > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
                optimizer.step()
                optimizer.zero_grad()
                accum_step = 0
        
        # Measure accuracy and record loss (using unscaled loss)
        with torch.no_grad():
            acc1 = accuracy(outputs, targets)[0]
        
        # Update metrics
        loss_meter.update(loss.item() * grad_accum_steps, targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc1_meter.avg:.2f}%',
            'time/img': f'{batch_time.avg:.3f}s',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Periodically clear CUDA cache to prevent fragmentation
        if idx > 0 and idx % 50 == 0:
            torch.cuda.empty_cache()
    
    # Log epoch stats
    epoch_time = time.time() - start_time
    logger.info(f"Epoch {epoch} - Train Loss: {loss_meter.avg:.4f}, Acc: {acc1_meter.avg:.2f}%, "
          f"Time: {datetime.timedelta(seconds=int(epoch_time))}")
    
    return loss_meter.avg, acc1_meter.avg


@torch.no_grad()
def validate(model, val_loader, criterion, device, logger, use_amp=False):
    """Optimized validation function"""
    model.eval()
    
    # Setup metrics
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    
    # Create progress bar
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    
    for inputs, targets in pbar:
        # Move data to device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass with mixed precision if enabled
        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Measure accuracy
        acc1 = accuracy(outputs, targets)[0]
        
        # Update metrics
        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc1_meter.avg:.2f}%'
        })
    
    logger.info(f"Validation - Loss: {loss_meter.avg:.4f}, Acc: {acc1_meter.avg:.2f}%")
    return loss_meter.avg, acc1_meter.avg


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    # Load config
    config = get_config(args)
    
    # Override config with command line arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.TRAIN.EPOCHS = args.epochs
    if args.lr:
        config.TRAIN.LR = args.lr
    if args.output:
        config.OUTPUT = args.output
    
    # Add additional parameters to config
    config.use_amp = not args.no_amp
    config.use_checkpoint = not args.no_checkpoint
    config.validate_freq = args.validate_freq
    config.grad_accum_steps = args.grad_accum_steps
    config.grad_clip_val = getattr(config.MEMORY, 'GRAD_CLIP_NORM', 1.0) if hasattr(config, 'MEMORY') else 1.0
    config.epochs = config.TRAIN.EPOCHS
    
    # Ensure output directory includes model type
    if config.OUTPUT:
        model_dir = f"{args.model_type}_{args.model_variant}"
        config.OUTPUT = os.path.join(config.OUTPUT, model_dir)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{args.model_type}_{args.model_variant}")
    
    # Save config
    config_path = os.path.join(config.OUTPUT, "config.yaml")
    with open(config_path, "w") as f:
        f.write(config.dump())
    logger.info(f"Config saved to {config_path}")
    
    # Build data loaders with optimized settings
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(
        config
    )
    
    # Optimize data loaders
    data_loader_train.pin_memory = True
    data_loader_val.pin_memory = True
    
    # Build the model
    model = build_model(args, config)
    model = model.to(device)
    
    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model_type}_{args.model_variant}, Parameters: {n_parameters/1e6:.2f}M")
    
    # Create optimizer
    if config.TRAIN.OPTIMIZER.NAME.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=getattr(config.TRAIN.OPTIMIZER, 'NESTEROV', True)
        )
    elif config.TRAIN.OPTIMIZER.NAME.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    elif config.TRAIN.OPTIMIZER.NAME.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.TRAIN.OPTIMIZER.NAME}")
    
    # Create learning rate scheduler
    min_lr = config.TRAIN.MIN_LR if hasattr(config.TRAIN, 'MIN_LR') else 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.TRAIN.EPOCHS,
        eta_min=min_lr
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if config.use_amp else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0)
            logger.info(f"Resumed from epoch {start_epoch-1} with accuracy {best_acc:.2f}%")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting from scratch")
    
    # Evaluate only if specified
    if args.eval:
        val_loss, val_acc = validate(model, data_loader_val, criterion, device, logger, config.use_amp)
        logger.info(f"Evaluation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        return
    
    # Training loop
    logger.info(f"Starting training for {config.epochs} epochs")
    logger.info(f"Batch size: {config.DATA.BATCH_SIZE}, Gradient accumulation steps: {config.grad_accum_steps}")
    logger.info(f"Effective batch size: {config.DATA.BATCH_SIZE * config.grad_accum_steps}")
    logger.info(f"Mixed precision: {config.use_amp}, Gradient checkpointing: {config.use_checkpoint}")
    
    # Get save frequency from config
    save_freq = config.SAVE_FREQ if hasattr(config, 'SAVE_FREQ') else 10
    if hasattr(config, 'MEMORY') and hasattr(config.MEMORY, 'SAVE_FREQ'):
        save_freq = config.MEMORY.SAVE_FREQ
    
    for epoch in range(start_epoch, config.epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, data_loader_train, criterion, optimizer, epoch, device, config, logger, scaler
        )
        
        # Validate if needed
        if epoch % config.validate_freq == 0 or epoch == config.epochs - 1:
            val_loss, val_acc = validate(model, data_loader_val, criterion, device, logger, config.use_amp)
            
            # Update best accuracy
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            logger.info(f"Best accuracy: {best_acc:.2f}%")
        else:
            val_loss, val_acc = -1, -1
            is_best = False
            logger.info(f"Skipping validation for epoch {epoch}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if epoch % save_freq == 0 or epoch == config.epochs - 1 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'config': config
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(config.OUTPUT, f"checkpoint_epoch{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Save best model if needed
            if is_best:
                best_path = os.path.join(config.OUTPUT, "model_best.pth")
                torch.save(checkpoint, best_path)
                logger.info(f"Best model saved to {best_path}")
        
        # Log metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        }
        
        # Save metrics to file
        metrics_path = os.path.join(config.OUTPUT, "metrics.json")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main() 