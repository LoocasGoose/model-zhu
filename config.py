# --------------------------------------------------------
# Inspired by Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Originally written by Ze Liu
# --------------------------------------------------------'

import os 
  
  
import yaml
from yacs.config import CfgNode as CN

# This is the config file that we will modify for each experiment
base_config = CN()

# Base config files to inherit from, relative to the current config file
base_config.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
base_config.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
base_config.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
base_config.DATA.DATA_PATH = ""
# Path specifically for Medium ImageNet dataset HDF5 file
base_config.DATA.MEDIUM_IMAGENET_PATH = ""
# Dataset name
base_config.DATA.DATASET = "cifar"
# Input image size
base_config.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
base_config.DATA.INTERPOLATION = "bicubic"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
base_config.DATA.PIN_MEMORY = True
# Number of data loading threads
base_config.DATA.NUM_WORKERS = 8
# Fraction of training data to use (1.0 means use all data, 0.1 means use 10% of the data)
base_config.DATA.SUBSET_FRACTION = 1.0
# Prefetch factor for data loading
base_config.DATA.PREFETCH_FACTOR = 2
# Whether to use persistent workers
base_config.DATA.PERSISTENT_WORKERS = True

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
base_config.MODEL = CN()
# Model NAME
base_config.MODEL.NAME = "swin"
# Checkpoint to resume, could be overwritten by command line argument
base_config.MODEL.RESUME = ""
# Number of classes, overwritten in data preparation
base_config.MODEL.NUM_CLASSES = 1000
# Dropout rate
base_config.MODEL.DROP_RATE = 0.0
# Whether to use small inputs architecture (for CIFAR-sized inputs)
base_config.MODEL.SMALL_INPUTS = True

# Resnet Transformer parameters
base_config.MODEL.RESNET = CN()

# ResNeXt parameters
base_config.MODEL.RESNEXT = CN()
# Cardinality - Number of transformation groups
base_config.MODEL.RESNEXT.CARDINALITY = 32
# Base width for each group
base_config.MODEL.RESNEXT.BASE_WIDTH = 4
# Channel pruning rate (1.0 means no pruning)
base_config.MODEL.RESNEXT.PRUNING_RATE = 1.0
# Activation function to use (relu, relu6, silu)
base_config.MODEL.RESNEXT.ACTIVATION = "relu"
# Whether to use gradient checkpointing
base_config.MODEL.RESNEXT.USE_CHECKPOINT = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
base_config.TRAIN = CN()
base_config.TRAIN.START_EPOCH = 0
base_config.TRAIN.EPOCHS = 300
base_config.TRAIN.WARMUP_EPOCHS = 20
base_config.TRAIN.LR = 5e-4
base_config.TRAIN.MIN_LR = 5e-4
base_config.TRAIN.WARMUP_LR = 5e-4

# Gradient accumulation steps
# could be overwritten by command line argument
base_config.TRAIN.ACCUMULATION_STEPS = 1
# Gradient accumulation steps for larger effective batch size
base_config.TRAIN.GRADIENT_ACCUMULATION_STEPS = 1
# Whether to use automatic mixed precision training
base_config.TRAIN.USE_AMP = False
# Mixed precision optimization level
base_config.TRAIN.OPT_LEVEL = "O1"

# LR scheduler
base_config.TRAIN.LR_SCHEDULER = CN()
base_config.TRAIN.LR_SCHEDULER.NAME = "cosine"

# Optimizer
base_config.TRAIN.OPTIMIZER = CN()
base_config.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
base_config.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
base_config.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
base_config.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
base_config.AUG = CN()
# Color jitter factor
base_config.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
base_config.AUG.RAND_AUGMENT = "rand-m9-mstd0.5-inc1"

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
base_config.TEST = CN()
# Whether to use center crop when testing
base_config.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
base_config.TEST.SEQUENTIAL = False
base_config.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Path to output folder, overwritten by command line argument
base_config.OUTPUT = ""
# Frequency to save checkpoint
base_config.SAVE_FREQ = 1
# Frequency to logging info
base_config.PRINT_FREQ = 10
# Fixed random seed
base_config.SEED = 0
# Perform evaluation only, overwritten by command line argument
base_config.EVAL_MODE = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Use the config in BASE as the default
    for base_cfg in yaml_cfg.setdefault("BASE", [""]):
        if base_cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), base_cfg))
    print(f"=> merge config from {cfg_file}")
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name):
            # Special handling for subset_fraction to allow 0.0 value
            if name == "subset_fraction":
                return args.subset_fraction is not None
            # For other arguments, use the original check
            else:
                return eval(f"args.{name}")
        return False

    # merge from specific arguments
    if _check_args("batch_size"):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args("data_path"):
        config.DATA.DATA_PATH = args.data_path
    if _check_args("resume"):
        config.MODEL.RESUME = args.resume
    if _check_args("use_checkpoint"):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args("output"):
        config.OUTPUT = args.output
    if _check_args("eval"):
        config.EVAL_MODE = True
    if _check_args("subset_fraction"):
        config.DATA.SUBSET_FRACTION = args.subset_fraction

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = base_config.clone()
    update_config(config, args)

    return config
