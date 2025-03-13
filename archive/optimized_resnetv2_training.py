#!/usr/bin/env python3
"""
Memory-optimized training script for ResNetV2.
Includes all optimizations to reduce memory usage and increase batch size.
Uses literature-recommended hyperparameters for best results.
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.resnetv2 import (
    resnet18, resnet34, resnet50, 
    mixup_data, cutmix_data, label_smoothing_loss
)
from data.datasets import MediumImagenetHDF5Dataset

# Default configuration
DEFAULT_CONFIG = {
    "DATA": {
        "BATCH_SIZE": 128,  # Increased batch size due to our optimizations
        "DATASET": "medium_imagenet",
        "MEDIUM_IMAGENET_PATH": '/honey/nmep/medium-imagenet-96.hdf5',
        "IMG_SIZE": 96,
        "NUM_WORKERS": 8,  # Adjust based on CPU cores
        "PIN_MEMORY": True,
        "PERSISTENT_WORKERS": True,
        "PREFETCH_FACTOR": 2
    },
    "MODEL": {
        "VARIANT": "resnet18",  # Options: resnet18, resnet34, resnet50
        "NUM_CLASSES": 200,
        "USE_SE": True,         # Still use SE blocks but memory-optimized
        "DROP_RATE": 0.15       # Moderate dropout
    },
    "TRAIN": {
        "EPOCHS": 60,           # Literature suggests 60-100 epochs
        "WARMUP_EPOCHS": 5,     # 5-10% of total epochs for warmup
        "LR": 0.01,             # Literature-recommended LR for SGD
        "MIN_LR": 0.0001,       # Minimum LR at end of schedule
        "OPTIMIZER": {
            "NAME": "sgd",      # SGD with momentum is literature standard
            "MOMENTUM": 0.9,    # Standard momentum value
            "WEIGHT_DECAY": 0.0001  # Literature-recommended weight decay
        },
        "LR_SCHEDULER": {
            "NAME": "cosine"    # Cosine with warmup is recommended
        },
        "USE_AMP": True,        # Use mixed precision
        "GRADIENT_CLIP_VAL": 1.0,  # Clip gradients for stability
        "GRADIENT_ACCUMULATION_STEPS": 1  # Can increase for larger effective batch
    },
    "AUG": {
        "MIXUP": 0.2,           # Reduced from original to save memory
        "CUTMIX": 0.1,          # Reduced from original to save memory
        "LABEL_SMOOTHING": 0.1  # Literature-standard value
    },
    "OUTPUT": "output/resnetv2_optimized",
    "SAVE_FREQ": 5,             # Save checkpoints every 5 epochs
    "PRINT_FREQ": 50            # Print metrics every 50 iterations
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Memory-optimized ResNetV2 training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model variant: resnet18, resnet34, resnet50 (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--disable-amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--disable-se', action='store_true',
                        help='Disable Squeeze-and-Excitation blocks')
    parser.add_argument('--disable-augment', action='store_true',
                        help='Disable mixup/cutmix augmentation')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate only')
    return parser.parse_args()


def load_config(args):
    """Load and update configuration"""
    config = DEFAULT_CONFIG.copy()
    
    # Load from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            # Deep update the config
            for section, items in file_config.items():
                if section in config and isinstance(items, dict):
                    config[section].update(items)
                else:
                    config[section] = items
    
    # Override config with command line args
    if args.batch_size:
        config["DATA"]["BATCH_SIZE"] = args.batch_size
    if args.lr:
        config["TRAIN"]["LR"] = args.lr
    if args.epochs:
        config["TRAIN"]["EPOCHS"] = args.epochs
    if args.model:
        config["MODEL"]["VARIANT"] = args.model
    if args.output:
        config["OUTPUT"] = args.output
    if args.data_path:
        config["DATA"]["MEDIUM_IMAGENET_PATH"] = args.data_path
    if args.disable_amp:
        config["TRAIN"]["USE_AMP"] = False
    if args.disable_se:
        config["MODEL"]["USE_SE"] = False
    if args.disable_augment:
        config["AUG"]["MIXUP"] = 0.0
        config["AUG"]["CUTMIX"] = 0.0
    if args.workers:
        config["DATA"]["NUM_WORKERS"] = args.workers
    
    return config


def get_model(config):
    """Create model based on config"""
    variant = config["MODEL"]["VARIANT"]
    kwargs = {
        'num_classes': config["MODEL"]["NUM_CLASSES"],
        'use_se': config["MODEL"]["USE_SE"],
        'dropout_rate': config["MODEL"]["DROP_RATE"],
        'zero_init_residual': True  # Literature recommendation
    }
    
    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50
    }
    
    if variant not in model_dict:
        raise ValueError(f"Unsupported model variant: {variant}")
        
    return model_dict[variant](**kwargs)


def get_optimizer(config, model):
    """Create optimizer based on config"""
    optimizer_name = config["TRAIN"]["OPTIMIZER"]["NAME"]
    lr = config["TRAIN"]["LR"]
    
    if optimizer_name == 'sgd':
        momentum = config["TRAIN"]["OPTIMIZER"]["MOMENTUM"]
        weight_decay = config["TRAIN"]["OPTIMIZER"]["WEIGHT_DECAY"]
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
            nesterov=True  # Literature recommendation
        )
    elif optimizer_name == 'adam':
        weight_decay = config["TRAIN"]["OPTIMIZER"]["WEIGHT_DECAY"]
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        weight_decay = config["TRAIN"]["OPTIMIZER"]["WEIGHT_DECAY"]
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(config, optimizer):
    """Create learning rate scheduler"""
    scheduler_name = config["TRAIN"]["LR_SCHEDULER"]["NAME"]
    epochs = config["TRAIN"]["EPOCHS"]
    warmup = config["TRAIN"]["WARMUP_EPOCHS"]
    min_lr = config["TRAIN"]["MIN_LR"]
    
    if scheduler_name == 'cosine':
        # Warmup + cosine decay - literature recommended
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["TRAIN"]["LR"],
            steps_per_epoch=1,  # We'll step once per epoch
            epochs=epochs,
            pct_start=warmup/epochs,  # Warmup period as percentage
            div_factor=10,  # LR will start from max_lr/10
            final_div_factor=config["TRAIN"]["LR"]/min_lr,  # End with min_lr
            anneal_strategy='cos'  # Cosine decay
        )
    elif scheduler_name == 'step':
        step_size = epochs // 3  # Typical step size is 1/3 of training
        gamma = 0.1  # Standard decay factor
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
    else:
        # Default to cosine schedule if not recognized
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs, 
            eta_min=min_lr
        )


def train_one_epoch(model, train_loader, optimizer, config, epoch, device, scaler=None):
    """Train the model for one epoch with memory optimizations"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Get augmentation parameters
    use_mixup = config["AUG"]["MIXUP"] > 0
    use_cutmix = config["AUG"]["CUTMIX"] > 0
    mixup_prob = min(0.5, config["AUG"]["MIXUP"])  # Limit to 50% max
    cutmix_prob = min(0.3, config["AUG"]["CUTMIX"])  # Limit to 30% max
    label_smoothing = config["AUG"]["LABEL_SMOOTHING"]
    
    use_amp = config["TRAIN"]["USE_AMP"]
    grad_accumulation = config["TRAIN"]["GRADIENT_ACCUMULATION_STEPS"]
    grad_clip = config["TRAIN"]["GRADIENT_CLIP_VAL"]
    
    # Create tqdm progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['TRAIN']['EPOCHS']-1}")
    
    # Set gradient accumulation counters
    optimizer.zero_grad()
    accumulation_steps = 0
    
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply augmentation with reduced probability
        rand_val = torch.rand(1).item()
        use_mix = None
        
        if use_mixup and rand_val < mixup_prob:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 0.8)  # Alpha = 0.8
            use_mix = 'mixup'
        elif use_cutmix and rand_val < mixup_prob + cutmix_prob:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, 1.0)  # Alpha = 1.0
            use_mix = 'cutmix'
        
        # Use mixed precision for forward pass
        if use_amp:
            with autocast():
                outputs = model(inputs)
                
                if use_mix:
                    loss = lam * label_smoothing_loss(outputs, targets_a, label_smoothing) + \
                        (1 - lam) * label_smoothing_loss(outputs, targets_b, label_smoothing)
                else:
                    loss = label_smoothing_loss(outputs, targets, label_smoothing)
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accumulation
                
            # Backward pass with scaler
            scaler.scale(loss).backward()
            accumulation_steps += 1
            
            # Step optimizer on accumulation boundary or at the end
            if accumulation_steps == grad_accumulation or i == len(train_loader) - 1:
                if grad_clip > 0:
                    # Unscale before clipping gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
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
            loss = loss / grad_accumulation
            
            # Backward pass
            loss.backward()
            accumulation_steps += 1
            
            # Step optimizer on accumulation boundary or at the end
            if accumulation_steps == grad_accumulation or i == len(train_loader) - 1:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                accumulation_steps = 0
        
        # Record stats
        total_loss += loss.item() * grad_accumulation  # Scale back the loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # For mixup, accuracy is approximate
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        accuracy = 100.0 * correct / total
        pbar.set_postfix({
            'loss': f'{total_loss/(i+1):.3f}',
            'acc': f'{accuracy:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Explicit memory cleanup every 50 batches
        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss / len(train_loader), accuracy


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Use tqdm for progress tracking
    pbar = tqdm(val_loader, desc="Validating")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
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
    
    return total_loss / len(val_loader), accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, config, is_best=False):
    """Save model checkpoint"""
    checkpoint_dir = os.path.join(config["OUTPUT"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save to the specific epoch checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'accuracy': accuracy,
        'config': config
    }, checkpoint_path)
    
    # If best model so far, create a copy
    if is_best:
        best_path = os.path.join(config["OUTPUT"], "model_best.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'accuracy': accuracy,
            'config': config
        }, best_path)
        print(f"Saved best model with accuracy {accuracy:.2f}% to {best_path}")
    
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main function to run training"""
    args = parse_args()
    config = load_config(args)
    
    # Create output directory
    os.makedirs(config["OUTPUT"], exist_ok=True)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Save config to output directory
    with open(os.path.join(config["OUTPUT"], 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    
    # Create model
    model = get_model(config)
    model = model.to(device)
    print(f"Created {config['MODEL']['VARIANT']} model")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dataloaders
    img_size = config["DATA"]["IMG_SIZE"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    train_dataset = MediumImagenetHDF5Dataset(
        img_size=img_size, 
        split='train',
        filepath=config["DATA"]["MEDIUM_IMAGENET_PATH"], 
        augment=True
    )
    val_dataset = MediumImagenetHDF5Dataset(
        img_size=img_size, 
        split='val',
        filepath=config["DATA"]["MEDIUM_IMAGENET_PATH"], 
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["DATA"]["NUM_WORKERS"],
        pin_memory=config["DATA"]["PIN_MEMORY"],
        persistent_workers=config["DATA"]["PERSISTENT_WORKERS"],
        prefetch_factor=config["DATA"]["PREFETCH_FACTOR"],
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=config["DATA"]["NUM_WORKERS"],
        pin_memory=config["DATA"]["PIN_MEMORY"],
        persistent_workers=config["DATA"]["PERSISTENT_WORKERS"],
        prefetch_factor=config["DATA"]["PREFETCH_FACTOR"]
    )
    
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config["TRAIN"]["USE_AMP"] else None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_accuracy = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('accuracy', 0)
        print(f"Resumed from epoch {start_epoch-1} with accuracy {best_accuracy:.2f}%")
    
    # Evaluate only if specified
    if args.eval:
        loss, accuracy = validate(model, val_loader, device)
        print(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        return
    
    # Train model
    print(f"Starting training for {config['TRAIN']['EPOCHS']} epochs")
    for epoch in range(start_epoch, config["TRAIN"]["EPOCHS"]):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, config, epoch, device, scaler
        )
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_acc > best_accuracy
        best_accuracy = max(val_acc, best_accuracy)
        
        if (epoch + 1) % config["SAVE_FREQ"] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, 
                epoch, val_acc, config, is_best
            )
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main() 