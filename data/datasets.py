from __future__ import annotations

import h5py
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
import os
import numpy as np
import torch.utils.data as data
from torchvision import datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True


# from torchvision.transforms import Compose, Normalize, PILToTensor, Resize


# deprecated in favor of MediumImagenetHDF5Dataset
# class MediumImagenetDataset(Dataset):
#     def __init__(self, img_size, split:str='train', augment=True):
#         assert split in ['train', 'val', 'test']
#         self.split = split
#         self.augment = augment
#         self.input_size = img_size
#         self.transform = self._get_transforms()
#         ds = ImageFolder("/data/medium-imagenet/data")
#         if split == 'train':
#             self.dataset = Subset(ds, range(0, len(ds) * 9 // 10))
#         else:
#             self.dataset = Subset(ds, range(len(ds) * 9 // 10, len(ds)))

#     def __getitem__(self, index):
#         image, label = self.dataset[index]
#         image = self.transform(image)
#         return image, label

#     def __len__(self):
#         return len(self.dataset)

#     def _get_transforms(self):
#         transform = []
#         transform.append(transforms.PILToTensor())
#         transform.append(lambda x: x.to(torch.float))
#         normalization = torch.Tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
#         transform.append(transforms.Normalize(normalization[0], normalization[1]))
#         if self.train and self.augment:
#             transform.extend(
#                 [
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#                 ]
#             )
#         transform.append(transforms.Resize([self.input_size] * 2))
#         return transforms.Compose(transform)


class MediumImagenetHDF5Dataset(Dataset):
    def __init__(
        self,
        img_size,
        split: str = "train",
        filepath: str | None = None,
        augment: bool = True,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.augment = augment
        self.input_size = img_size
        self.transform = self._get_transforms()
        
        # Use the provided filepath or default to /data/imagenet
        if filepath is None or filepath == "":
            # Default path for Medium ImageNet on honeydew
            default_path = "/data/imagenet/medium-imagenet-nmep-96.hdf5"
            try:
                self.file = h5py.File(default_path, "r")
                print(f"Successfully loaded dataset from {default_path}")
            except (IOError, OSError) as e:
                raise FileNotFoundError(
                    f"Could not find the Medium ImageNet dataset at {default_path}. "
                    "Please specify the correct path using the DATA.MEDIUM_IMAGENET_PATH config option."
                )
        else:
            self.file = h5py.File(filepath, "r")

    def __getitem__(self, index):
        image = self.file[f"images-{self.split}"][index]
        if self.split != "test":
            label = self.file[f"labels-{self.split}"][index]
        else:
            label = -1
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.file[f"images-{self.split}"])

    def _get_transforms(self):
        transform = []
        transform.append(lambda x: torch.tensor(x / 256).to(torch.float))
        normalization = torch.Tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform.append(transforms.Normalize(normalization[0], normalization[1]))
        transform.append(transforms.Resize([self.input_size] * 2))
        if self.split == "train" and self.augment:
            transform.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                ]
            )
        return transforms.Compose(transform)


class CIFAR10Dataset(Dataset):
    def __init__(self, img_size=32, train=True):
        self.train = train
        self.img_size = img_size

        # Build optimized transform pipeline
        self.transform = self._get_transforms()
        self.dataset = CIFAR10(root="/data/cifar10", train=self.train, download=True)
        
        # Pre-normalize mean and std for efficiency
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

    def _get_transforms(self):
        # CIFAR-10 images are already 32x32, so only resize if needed
        need_resize = self.img_size != 32
        
        if self.train:
            # More efficient transform pipeline for training
            if need_resize:
                transform = [
                    # Convert to tensor first for faster operations
                    transforms.ToTensor(),
                    # Apply color jitter directly on tensor for efficiency
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    # Normalize before resize to reduce computation on smaller images
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    transforms.Resize([self.img_size, self.img_size]),
                ]
            else:
                # Even more efficient when no resize is needed
                transform = [
                    transforms.ToTensor(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
        else:
            # Validation/test pipeline - simpler and more efficient
            if need_resize:
                transform = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    transforms.Resize([self.img_size, self.img_size]),
                ]
            else:
                transform = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
        
        return transforms.Compose(transform)


class MediumImageNetDataset(data.Dataset):
    """
    Dataset class for Medium ImageNet dataset stored in HDF5 format.
    
    Args:
        file_path (str): Path to the HDF5 file
        transform (callable, optional): Optional transform to be applied on a sample
        split (str): Which split to use ('train', 'val', or 'test')
    """
    def __init__(self, file_path, transform=None, split='train'):
        self.file_path = file_path
        self.transform = transform
        self.split = split
        
        # Open the HDF5 file for inspection (don't keep it open)
        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f[f'{split}_labels'])
            
        # Class names will be initialized on first access
        self._class_names = None
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Open file in read mode
        with h5py.File(self.file_path, 'r') as f:
            # Read image data as a numpy array
            img = f[f'{self.split}_images'][idx]
            # Read label
            label = f[f'{self.split}_labels'][idx]
            
        # Convert numpy array to PIL Image
        img = Image.fromarray(img)
        
        # Apply transformations if available
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    @property
    def class_names(self):
        # Lazy loading of class names
        if self._class_names is None:
            with h5py.File(self.file_path, 'r') as f:
                # Assuming class names are stored in the HDF5 file
                # If not, you can load them from elsewhere
                if 'class_names' in f:
                    self._class_names = [name.decode('utf-8') for name in f['class_names'][:]]
                else:
                    # Default to generic class names if not available
                    self._class_names = [f'Class {i}' for i in range(200)]
        
        return self._class_names


def build_transform(is_train, img_size, color_jitter=0.4):
    """
    Build transformation pipeline for training or validation.
    
    Args:
        is_train (bool): Whether to build transformations for training
        img_size (int): Size of the input image
        color_jitter (float): Strength of color jittering
        
    Returns:
        transform (callable): Transformation pipeline
    """
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter/2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),  # 14% larger for validation
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def build_dataset(config, is_train):
    """
    Build dataset based on config.
    
    Args:
        config (dict): Configuration dictionary
        is_train (bool): Whether to build dataset for training
        
    Returns:
        dataset (Dataset): Dataset object
    """
    img_size = config.DATA.IMG_SIZE
    color_jitter = config.AUG.COLOR_JITTER if hasattr(config.AUG, 'COLOR_JITTER') else 0.4
    transform = build_transform(is_train, img_size, color_jitter)
    
    if config.DATA.DATASET == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data',
            train=is_train,
            download=True,
            transform=transform
        )
    elif config.DATA.DATASET == 'cifar100':
        dataset = datasets.CIFAR100(
            root='./data',
            train=is_train,
            download=True,
            transform=transform
        )
    elif config.DATA.DATASET == 'medium_imagenet':
        split = 'train' if is_train else 'val'
        dataset = MediumImageNetDataset(
            file_path=config.DATA.MEDIUM_IMAGENET_PATH,
            transform=transform,
            split=split
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.DATA.DATASET}")
    
    return dataset


def build_dataloader(config, is_train):
    """
    Build dataloader based on config.
    
    Args:
        config (dict): Configuration dictionary
        is_train (bool): Whether to build dataloader for training
        
    Returns:
        dataloader (DataLoader): DataLoader object
    """
    dataset = build_dataset(config, is_train)
    
    if is_train:
        batch_size = config.DATA.BATCH_SIZE
        shuffle = True
    else:
        batch_size = 2 * config.DATA.BATCH_SIZE
        shuffle = False
    
    num_workers = config.DATA.NUM_WORKERS if hasattr(config.DATA, 'NUM_WORKERS') else 4
    pin_memory = config.DATA.PIN_MEMORY if hasattr(config.DATA, 'PIN_MEMORY') else False
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train
    )
    
    return dataloader
