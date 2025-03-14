import argparse
import datetime
import json
import os
import shutil
import time
import contextlib  # For nullcontext

import numpy as np
import torch
import torch.nn as nn
from timm.utils.metrics import AverageMeter, accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset  # For custom datasets
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_str
from torch.cuda.amp import GradScaler, autocast  # Import for mixed precision training

from config import get_config
from data import build_loader
from models import build_model
from optimizer import build_optimizer
from utils import create_logger, load_checkpoint, save_checkpoint


def parse_option():
    parser = argparse.ArgumentParser("Vision model training and evaluation script", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--subset-fraction", type=float, default=1.0, help="fraction of training data to use")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--enable-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--validate-freq", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Number of batches loaded in advance by dataloader workers")
    parser.add_argument("--pin-memory", action="store_true", help="Pin memory in dataloader for potentially faster data transfer")

    args = parser.parse_args()

    config = get_config(args)
    
    # Defrost the config to make it mutable before modifying
    config.defrost()
    
    # Ensure command line args are properly set in config
    # Mixed precision training
    if args.use_amp:
        if not hasattr(config.TRAIN, 'USE_AMP'):
            # If the attribute doesn't exist in config, add it
            setattr(config.TRAIN, 'USE_AMP', True)
        else:
            # Otherwise override it
            config.TRAIN.USE_AMP = True
            
    # Set validate frequency from command line args
    if args.validate_freq > 1:
        setattr(config, 'VALIDATE_FREQ', args.validate_freq)
        
    # Set gradient checkpointing from command line args
    if args.enable_checkpoint:
        if not hasattr(config.MODEL, 'ENABLE_CHECKPOINT'):
            setattr(config.MODEL, 'ENABLE_CHECKPOINT', True)
        else:
            config.MODEL.ENABLE_CHECKPOINT = True
            
    # Data loading optimizations
    if args.pin_memory:
        config.DATA.PIN_MEMORY = True
        
    if args.prefetch_factor != 2:  # Only if different from default
        if not hasattr(config.DATA, 'PREFETCH_FACTOR'):
            setattr(config.DATA, 'PREFETCH_FACTOR', args.prefetch_factor)
        else:
            config.DATA.PREFETCH_FACTOR = args.prefetch_factor
    
    # Freeze the config again to make it immutable
    config.freeze()

    return args, config


def main(config):
    # Enable cuDNN benchmark for faster performance and set memory allocation to be more efficient
    torch.backends.cudnn.benchmark = True
    # Allow TF32 on Ampere GPUs for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(
        config
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config)
    # logger.info(str(model))

    # Move model to device
    model = model.to(device)
    logger.info(f"Model moved to {device}")

    # param and flop counts
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    toy_input = torch.rand(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE).to(device)
    flops = FlopCountAnalysis(model, toy_input)
    del toy_input

    # print("Model = %s" % str(model_without_ddp))
    n_flops = flops.total()
    logger.info(flop_count_str(flops))
    logger.info('number of params: {} M'.format(n_parameters / 1e6))
    logger.info(f'flops: {n_flops/1e6} MFLOPS')

    # Keep it simple with basic epoch scheduler
    optimizer = build_optimizer(config, model)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS)

    # Initialize gradient scaler for mixed precision training
    use_amp = getattr(config.TRAIN, 'USE_AMP', False)
    scaler = GradScaler() if use_amp else None
    logger.info(f"Using mixed precision training: {use_amp}")
    
    # Get validation frequency
    validate_freq = getattr(config, 'VALIDATE_FREQ', 1)
    logger.info(f"Validating every {validate_freq} epochs")

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        acc1, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} val images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_acc1, train_loss = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, scaler)
        logger.info(f" * Train Acc {train_acc1:.3f} Train Loss {train_loss:.3f}")
        logger.info(f"Accuracy of the network on the {len(dataset_train)} train images: {train_acc1:.1f}%")

        # Only validate every validate_freq epochs to speed up training, except for the last epoch
        if epoch % validate_freq == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            val_acc1, val_loss = validate(config, data_loader_val, model)
            logger.info(f" * Val Acc {val_acc1:.3f} Val Loss {val_loss:.3f}")
            logger.info(f"Accuracy of the network on the {len(dataset_val)} val images: {val_acc1:.1f}%")
            
            # Update max_accuracy
            max_accuracy = max(max_accuracy, val_acc1)
            logger.info(f"Max accuracy: {max_accuracy:.2f}%\n")
        else:
            # Skip validation this epoch
            val_acc1, val_loss = -1, -1
            logger.info(f"Skipping validation for epoch {epoch} to speed up training")

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)

        lr_scheduler.step()

        # Clean up GPU memory after each epoch
        torch.cuda.empty_cache()

        log_stats = {"epoch": epoch, "n_params": n_parameters, "n_flops": n_flops,
                     "train_acc": train_acc1, "train_loss": train_loss, 
                     "val_acc": val_acc1, "val_loss": val_loss}
        with open(
                os.path.join(config.OUTPUT, "metrics.json"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))

    logger.info("Start testing")
    preds = evaluate(config, data_loader_test, model)
    np.save(os.path.join(config.OUTPUT, "preds.npy"), preds)
    # TODO save predictions to csv in kaggle format


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, scaler=None):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()  # Track data loading time
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    # Determine whether to use mixed precision
    use_amp = scaler is not None
    if use_amp:
        logger.info(f"Using mixed precision training for epoch {epoch}")
    
    # Get gradient clipping value if configured
    grad_clip_val = getattr(config.TRAIN, 'GRADIENT_CLIP_VAL', 0.0)
    
    # Wrap with tqdm progress bar - update less frequently for less overhead
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{config.TRAIN.EPOCHS}", ncols=100, leave=False)
    
    # Pre-fetch and pre-allocate next batch data
    optimizer.zero_grad(set_to_none=True)  # More efficient than standard zero_grad
    
    for idx, (samples, targets) in enumerate(pbar):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device with non_blocking for potential overlap
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        # Use automatic mixed precision for faster computation
        if use_amp:
            with autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
                
            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()
            
            # Apply gradient clipping if configured
            if grad_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # More efficient than standard zero_grad
        else:
            # Standard forward/backward pass without mixed precision
            outputs = model(samples)
            loss = criterion(outputs, targets)
            
            # Standard backward pass without mixed precision
            loss.backward()
            
            # Apply gradient clipping if configured
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # More efficient

        # Calculate accuracy
        with torch.no_grad():  # Ensure this doesn't add to computation graph
            (acc1,) = accuracy(outputs, targets)
            
        # Update metrics
        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Update progress bar less frequently to reduce overhead
        if idx % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc1_meter.avg:.2f}%',
                'amp': 'on' if use_amp else 'off',
                'time/img': f'{batch_time.avg:.3f}s'
            })
            
        # Periodically empty CUDA cache to prevent fragmentation (every 200 batches)
        if idx > 0 and idx % 200 == 0:
            torch.cuda.empty_cache()

    # Print final epoch stats
    epoch_time = time.time() - start
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f"EPOCH {epoch} training summary: "
        f"loss {loss_meter.avg:.4f}, "
        f"accuracy {acc1_meter.avg:.2f}%, "
        f"lr {optimizer.param_groups[0]['lr']:.6f}, "
        f"mem {memory_used:.0f}MB, "
        f"time {datetime.timedelta(seconds=int(epoch_time))}, "
        f"AMP: {'on' if use_amp else 'off'}"
    )
    
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    
    # Determine whether to use mixed precision
    use_amp = getattr(config.TRAIN, 'USE_AMP', False)
    if use_amp:
        logger.info("Using mixed precision for validation")

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output with mixed precision for efficiency
        if use_amp:
            with autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        (acc1,) = accuracy(output, target)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print progress less frequently for less overhead
        if idx % 20 == 0 or idx == len(data_loader) - 1:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Validate: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )

    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(f"Validation complete - Accuracy: {acc1_meter.avg:.2f}%, Loss: {loss_meter.avg:.4f}, Memory: {memory_used:.0f}MB, AMP: {'on' if use_amp else 'off'}")
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def evaluate(config, data_loader, model):
    model.eval()
    preds = []
    
    # Determine whether to use mixed precision
    use_amp = getattr(config.TRAIN, 'USE_AMP', False)
    if use_amp:
        logger.info("Using mixed precision for evaluation")
    
    # Use a small batch aggregation to reduce memory pressure
    batch_size = 10
    aggregated_outputs = []
    
    for idx, (images, _) in enumerate(tqdm(data_loader, desc="Evaluating")):
        images = images.cuda(non_blocking=True)
        
        # Use mixed precision for inference if enabled
        if use_amp:
            with autocast():
                output = model(images)
        else:
            output = model(images)
            
        aggregated_outputs.append(output.cpu().numpy())
        
        # Periodically concatenate and clear to save memory
        if len(aggregated_outputs) >= batch_size or idx == len(data_loader) - 1:
            preds.append(np.concatenate(aggregated_outputs))
            aggregated_outputs = []
            
            # Explicitly clear GPU memory
            if idx % 50 == 0:
                torch.cuda.empty_cache()
    
    preds = np.concatenate(preds)
    return preds


if __name__ == "__main__":
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Set faster performance flag - allow non-deterministic algorithms
    torch.backends.cudnn.benchmark = True  # This can significantly speed up training

    # Make output dir
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.yaml")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
