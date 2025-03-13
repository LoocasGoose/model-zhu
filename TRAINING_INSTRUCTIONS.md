# ResNet Training Instructions

This document provides instructions on how to use the ResNet training scripts to train and evaluate different ResNet model variants.

## Overview

The codebase provides a unified training script that can handle both original ResNet models (from `resnet.py`) and improved ResNetV2 models (from `resnetv2.py`), with all the memory and performance optimizations. Three main scripts are provided:

1. `train_unified_resnet.py` - The main training script that handles all model variants
2. `run_resnet_training.py` - Python script for running different training configurations
3. `run_training.bat` / `run_training.ps1` - Windows batch/PowerShell scripts for easy launching

## Quick Start

### Using Batch Files (Windows)

For Windows users, the easiest way to run the training is with the provided batch files:

```batch
# Using Command Prompt (CMD)
run_training.bat resnet18 0 output 64 10

# Using PowerShell
.\run_training.ps1 -Mode resnet18 -GPU 0 -OutputDir output -BatchSize 64 -Epochs 10
```

### Using Python Script

For more advanced configurations, you can use the Python training runner:

```bash
python run_resnet_training.py --mode resnet18 --gpu 0 --output output --batch-size 64 --epochs 10
```

## Available Training Modes

The following training modes are available:

- `resnet18` - Train the original ResNet18 from `resnet.py`
- `resnetv2_18` - Train the improved ResNet18 from `resnetv2.py`
- `resnetv2_50` - Train the improved ResNet50 from `resnetv2.py`
- `all` - Train all three model variants sequentially
- `eval` - Evaluate trained models
- `compare` - Run a fair comparison with identical settings for all models
- `memory_efficient` - Train models with memory optimization settings

## Examples

### Training Original ResNet18

```bash
python run_resnet_training.py --mode resnet18 --gpu 0 --output output
```

### Training ResNetV2-50

```bash
python run_resnet_training.py --mode resnetv2_50 --gpu 0 --output output
```

### Memory-Efficient Training

```bash
python run_resnet_training.py --mode memory_efficient --gpu 0 --output output
```

### Evaluating Trained Models

```bash
python run_resnet_training.py --mode eval --gpu 0 --output output
```

### Comparing All Models

```bash
python run_resnet_training.py --mode compare --gpu 0 --output output
```

## Advanced Configuration

For advanced users who want to directly use the unified training script:

```bash
python train_unified_resnet.py \
  --model-type resnetv2 \
  --model-variant resnet50 \
  --cfg configs/resnet50_imagenet.yaml \
  --batch-size 64 \
  --epochs 100 \
  --lr 0.1 \
  --output output/resnetv2_50 \
  --validate-freq 5 \
  --grad-accum-steps 2 \
  --gpu 0
```

## Command Line Arguments

### For `run_resnet_training.py`

- `--mode`: Training mode (see above)
- `--output`: Base output directory
- `--gpu`: GPU ID to use
- `--batch-size`: Override batch size
- `--epochs`: Override number of epochs
- `--config`: Override default config file

### For `train_unified_resnet.py`

- `--cfg`: Path to config file (required)
- `--model-type`: Model architecture type ("resnet" or "resnetv2")
- `--model-variant`: Model size/variant ("resnet18" or "resnet50")
- `--batch-size`: Batch size (overrides config)
- `--epochs`: Number of epochs (overrides config) 
- `--lr`: Learning rate (overrides config)
- `--output`: Output directory (overrides config)
- `--workers`: Number of data loading workers
- `--validate-freq`: Validate every N epochs
- `--grad-accum-steps`: Gradient accumulation steps
- `--no-amp`: Disable mixed precision training
- `--no-checkpoint`: Disable gradient checkpointing
- `--resume`: Resume from checkpoint
- `--eval`: Evaluate only
- `--seed`: Random seed
- `--gpu`: GPU ID to use

## Configuration Files

The training script uses YAML configuration files for model and training settings:

- `configs/resnet18_medium_imagenet.yaml` - For original ResNet18
- `configs/resnet18_imagenet.yaml` - For ResNetV2-18
- `configs/resnet50_imagenet.yaml` - For ResNetV2-50

You can modify these files to change learning rates, optimizers, batch sizes, etc. 