from torch.utils.data import DataLoader, Subset
import random

from data.datasets import CIFAR10Dataset, MediumImagenetHDF5Dataset


def build_loader(config):
    # Create full datasets
    if config.DATA.DATASET == "cifar10":
        dataset_train = CIFAR10Dataset(img_size=config.DATA.IMG_SIZE, train=True)
        dataset_val = CIFAR10Dataset(img_size=config.DATA.IMG_SIZE, train=False)
        dataset_test = CIFAR10Dataset(img_size=config.DATA.IMG_SIZE, train=False)
    elif config.DATA.DATASET == "medium_imagenet":
        # Get filepath from config - empty string will trigger the default path in the dataset class
        filepath = config.DATA.MEDIUM_IMAGENET_PATH
        
        dataset_train = MediumImagenetHDF5Dataset(
            config.DATA.IMG_SIZE,
            split="train",
            filepath=filepath,
            augment=True
        )
        dataset_val = MediumImagenetHDF5Dataset(
            config.DATA.IMG_SIZE,
            split="val",
            filepath=filepath,
            augment=False
        )
        dataset_test = MediumImagenetHDF5Dataset(
            config.DATA.IMG_SIZE,
            split="test",
            filepath=filepath,
            augment=False
        )
    else:
        raise NotImplementedError

    # Print total size of original dataset
    full_size = len(dataset_train)
    print(f"Full training dataset size: {full_size} samples")

    # Apply subsetting to training data if specified
    if hasattr(config.DATA, 'SUBSET_FRACTION') and 0.0 < config.DATA.SUBSET_FRACTION < 1.0:
        subset_size = int(full_size * config.DATA.SUBSET_FRACTION)
        if subset_size <= 0:
            subset_size = 1  # Ensure at least one sample is used
        
        # Use random indices for the subset
        indices = random.sample(range(full_size), subset_size)
        dataset_train = Subset(dataset_train, indices)
        print(f"Using {subset_size} samples ({config.DATA.SUBSET_FRACTION:.2%} of original dataset) for training")
    else:
        # Using full dataset
        print(f"Using full dataset ({full_size} samples) for training")

    # Determine if we can use persistent workers (only supported in PyTorch 1.7.0+)
    persistent_workers = getattr(config.DATA, 'PERSISTENT_WORKERS', True) and config.DATA.NUM_WORKERS > 0
    prefetch_factor = getattr(config.DATA, 'PREFETCH_FACTOR', 2) if config.DATA.NUM_WORKERS > 0 else None
    pin_memory = getattr(config.DATA, 'PIN_MEMORY', True)
    
    print(f"DataLoader settings: persistent_workers={persistent_workers}, "
          f"prefetch_factor={prefetch_factor}, pin_memory={pin_memory}")
    
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for consistent sizes
        persistent_workers=persistent_workers,  # Keep workers alive between epochs
        prefetch_factor=prefetch_factor,  # Prefetch data
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test
