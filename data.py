import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image
from timm.data.transforms_factory import create_transform
import io
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


class FastH5Dataset(Dataset):
    """
    Memory-efficient HDF5 dataset that uses optimized IO operations
    """
    def __init__(self, h5_path, transform=None, is_train=True):
        """
        Initialize dataset from HDF5 file.
        
        Args:
            h5_path: Path to HDF5 file
            transform: Torchvision transforms to apply
            is_train: Whether this is for training or not
        """
        self.h5_path = h5_path
        self.transform = transform
        self.is_train = is_train
        
        # Open file in read-only mode
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Determine dataset splits
        if is_train:
            self.images = self.h5_file['train_images']
            self.labels = self.h5_file['train_labels'][:]  # Load labels into memory as they're small
        else:
            self.images = self.h5_file['val_images']
            self.labels = self.h5_file['val_labels'][:]
        
        self.num_samples = len(self.labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get image data - keeping on disk until needed
        img_data = self.images[idx]
        label = self.labels[idx]
        
        # Efficient image decoding
        img = torch.from_numpy(img_data).permute(2, 0, 1).float() / 255.0
        
        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)
            
        return img, label


class MemoryEfficientImageNetDataset(Dataset):
    """
    Memory-efficient dataset for ImageNet-style folder structure
    """
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory of the dataset
            transform: Torchvision transforms to apply
            is_train: Whether this is for training or not
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Determine dataset split folder
        split_folder = 'train' if is_train else 'val'
        self.data_dir = os.path.join(root_dir, split_folder)
        
        # Get all class folders
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                              if os.path.isdir(os.path.join(self.data_dir, d))])
        
        # Create class-to-index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Create samples list (filepath, class_idx)
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        """Create list of (sample path, class_idx) tuples"""
        samples = []
        for target_class in self.classes:
            class_idx = self.class_to_idx[target_class]
            class_dir = os.path.join(self.data_dir, target_class)
            
            for root, _, filenames in os.walk(class_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(root, filename)
                        samples.append((path, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        path, target = self.samples[idx]
        
        # Efficient image loading with Pillow
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            
            # Apply transforms if provided
            if self.transform:
                img = self.transform(img)
            
        return img, target


def build_transform(is_train, config):
    """
    Build optimized transformation pipeline
    
    Args:
        is_train: Whether this is for training or not
        config: Configuration object
        
    Returns:
        Transformation pipeline
    """
    img_size = config.DATA.IMG_SIZE
    
    if is_train:
        # Training transforms with efficient augmentations
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Add RandAugment if specified in config
        if hasattr(config.AUG, 'RANDAUG') and config.AUG.RANDAUG:
            from timm.data.auto_augment import rand_augment_transform
            transform.transforms.insert(2, rand_augment_transform(
                config.AUG.RANDAUG.NUM_OPS, 
                config.AUG.RANDAUG.MAGNITUDE
            ))
    else:
        # Validation/testing transforms (simpler and faster)
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    return transform


def build_loader(config):
    """
    Build optimized data loaders
    
    Args:
        config: Configuration object
        
    Returns:
        Training, validation, and test datasets and data loaders
    """
    # Create transforms
    transform_train = build_transform(is_train=True, config=config)
    transform_test = build_transform(is_train=False, config=config)
    
    # Choose dataset based on config
    if config.DATA.DATASET.lower() == 'medium_imagenet' and config.DATA.MEDIUM_IMAGENET_PATH.endswith('.hdf5'):
        # Use optimized HDF5 dataset for medium_imagenet
        dataset_train = FastH5Dataset(
            config.DATA.MEDIUM_IMAGENET_PATH, 
            transform=transform_train, 
            is_train=True
        )
        
        dataset_val = FastH5Dataset(
            config.DATA.MEDIUM_IMAGENET_PATH, 
            transform=transform_test, 
            is_train=False
        )
        
        # Test dataset is same as validation for medium_imagenet
        dataset_test = dataset_val
    else:
        # For standard image folder structure
        dataset_train = MemoryEfficientImageNetDataset(
            config.DATA.DATA_PATH,
            transform=transform_train,
            is_train=True
        )
        
        dataset_val = MemoryEfficientImageNetDataset(
            config.DATA.DATA_PATH,
            transform=transform_test,
            is_train=False
        )
        
        # Test dataset is same as validation 
        dataset_test = dataset_val
    
    # Create data loaders with optimized settings
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=getattr(config.DATA, 'PIN_MEMORY', True),  # Faster CPU->GPU transfers
        drop_last=True,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=getattr(config.DATA, 'PREFETCH_FACTOR', 2),  # Prefetch batches
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=getattr(config.DATA, 'PIN_MEMORY', True),
        drop_last=False,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False,
    )
    
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=getattr(config.DATA, 'PIN_MEMORY', True),
        drop_last=False,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False,
    )
    
    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test 