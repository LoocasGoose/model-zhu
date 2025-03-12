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

    # Apply subsetting to training data if specified
    if hasattr(config.DATA, 'SUBSET_FRACTION') and config.DATA.SUBSET_FRACTION < 1.0:
        subset_size = int(len(dataset_train) * config.DATA.SUBSET_FRACTION)
        if subset_size <= 0:
            subset_size = 1  # Ensure at least one sample is used
        
        # Use random indices for the subset
        indices = random.sample(range(len(dataset_train)), subset_size)
        dataset_train = Subset(dataset_train, indices)
        print(f"Using {subset_size} samples ({config.DATA.SUBSET_FRACTION:.2%} of original dataset) for training")

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=False,
    )

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=False,
    )

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test
