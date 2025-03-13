# ResNetV2 Training Pipeline

This repository contains an optimized implementation of ResNetV2 for image classification, with a focus on improving both speed and accuracy over the original ResNet implementation.

## Key Features

- **Multiple ResNet Variants**: Support for ResNet-18, 34, 50, 101, and 152
- **Squeeze-and-Excitation Blocks**: Optional SE blocks to boost representational power
- **Advanced Training Techniques**:
  - MixUp and CutMix data augmentation
  - Label smoothing
  - Advanced learning rate scheduling
  - Weight initialization optimization
- **Performance Improvements**:
  - Memory-efficient operations
  - Optimized forward pass
  - Bottleneck architecture for deeper networks

## Configuration

The training process is controlled through YAML configuration files. Two config files are provided:

1. `configs/resnetv2_base.yaml`: Base configuration with defaults for common settings
2. `configs/resnetv2_medium_imagenet.yaml`: Configuration for Medium ImageNet dataset that extends the base config

### Configuration Options

#### Model Configuration
```yaml
MODEL:
  NAME: "resnetv2"
  VARIANT: "resnet50"  # Options: resnet18, resnet34, resnet50, resnet101, resnet152
  NUM_CLASSES: 200
  DROP_RATE: 0.2  # Dropout probability
  ZERO_INIT_RESIDUAL: True  # Initialize residual branch BN to zero
  USE_SE: True  # Use Squeeze-and-Excitation blocks
```

#### Data Configuration
```yaml
DATA:
  BATCH_SIZE: 32
  DATASET: "medium_imagenet"
  MEDIUM_IMAGENET_PATH: '/path/to/medium-imagenet-96.hdf5'
  IMG_SIZE: 96
  NUM_WORKERS: 32
  PIN_MEMORY: True
```

#### Training Configuration
```yaml
TRAIN:
  EPOCHS: 30
  WARMUP_EPOCHS: 3
  LR: 5e-4
  MIN_LR: 5e-5
  LR_SCHEDULER:
    NAME: "warmup_cosine"  # Options: cosine, step, warmup_cosine, one_cycle
```

#### Augmentation Configuration
```yaml
AUG:
  COLOR_JITTER: 0.4
  MIXUP: 0.2  # MixUp alpha parameter
  CUTMIX: 0.2  # CutMix alpha parameter
  LABEL_SMOOTHING: 0.1  # Label smoothing factor
```

## Training

To train the model on Medium ImageNet:

```bash
python train_resnetv2.py --config configs/resnetv2_medium_imagenet.yaml
```

Additional command line options:
- `--resume /path/to/checkpoint.pth`: Resume training from a checkpoint
- `--output /path/to/output/dir`: Specify custom output directory
- `--gpu 0`: Select GPU device
- `--eval`: Run evaluation only (requires --resume to specify a model)
- `--seed 42`: Set random seed for reproducibility

## Performance Tips

1. **Use ResNet-50 or higher for complex datasets**: For Medium ImageNet, ResNet-50 with SE blocks provides a good balance of speed and accuracy.
2. **Enable augmentation**: MixUp, CutMix, and label smoothing significantly improve generalization.
3. **Learning rate scheduling**: The warmup_cosine scheduler works well for most scenarios.
4. **Batch size**: Use the largest batch size that fits in GPU memory for better throughput.
5. **Zero initialization**: Setting `ZERO_INIT_RESIDUAL: True` improves training stability.

## Differences from Original ResNet

The ResNetV2 implementation includes several improvements over the original ResNet:

1. **Squeeze-and-Excitation Blocks**: Recalibrate channel-wise feature responses adaptively.
2. **Improved Initialization**: Proper weight initialization for better convergence.
3. **Regularization Techniques**: Dropout and advanced data augmentation methods.
4. **Memory Efficiency**: Optimized operations to reduce memory footprint.
5. **Advanced Learning Rate Schedules**: Better convergence with warmup and cycling strategies.

## Requirements

- PyTorch >= 1.7.0
- torchvision
- PyYAML
- tensorboard
- h5py (for Medium ImageNet dataset)
- numpy 