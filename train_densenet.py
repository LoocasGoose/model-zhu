"""
DenseNet Training Script with Advanced Training Techniques

This script trains the DenseNet model on the medium-imagenet dataset
with techniques like early stopping, mixed precision training, and learning rate scheduling.
"""
import os
import time
import argparse
import logging
from typing import Dict, Tuple, Optional, List, Any, Union
from types import SimpleNamespace

from PIL.Image import _initialized
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Import the DenseNet model
from models.densenet import DenseNet121, DenseNet169, DenseNet201


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("densenet_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DenseNetTraining")


class HDF5Dataset(Dataset):
    """Dataset for loading images from HDF5 files"""
    
    def __init__(self, h5_path: str, transform=None):
        """
        Args:
            h5_path: Path to the HDF5 file
            transform: Optional transform to apply to the data
        """
        self.h5_path = h5_path
        self.transform = transform
        
        # Open the HDF5 file and get dataset sizes
        with h5py.File(self.h5_path, 'r') as h5_file:
            # Print available keys in the HDF5 file for debugging
            print(f"Available keys in HDF5 file: {list(h5_file.keys())}")
            
            # Try to identify image and label datasets
            self.images_key = None
            self.labels_key = None
            
            # Look for common naming patterns
            possible_image_keys = ['images', 'image', 'x', 'data', 'features']
            possible_label_keys = ['labels', 'label', 'y', 'targets', 'classes']
            
            # Check for image dataset
            for key in possible_image_keys:
                if key in h5_file and isinstance(h5_file[key], h5py.Dataset):
                    self.images_key = key
                    break
            
            # Check for label dataset
            for key in possible_label_keys:
                if key in h5_file and isinstance(h5_file[key], h5py.Dataset):
                    self.labels_key = key
                    break
            
            # If not found, try to infer from available keys
            if self.images_key is None or self.labels_key is None:
                for key in h5_file.keys():
                    if isinstance(h5_file[key], h5py.Dataset):
                        # Get shape safely
                        try:
                            shape = h5_file[key].shape
                            if len(shape) >= 3:  # Images typically have 3+ dimensions
                                self.images_key = key
                            elif len(shape) == 1 or len(shape) == 2:  # Labels typically have 1-2 dimensions
                                self.labels_key = key
                        except (AttributeError, TypeError):
                            continue
            
            # Verify we found the datasets
            if self.images_key is None:
                raise ValueError(f"Could not find image dataset in HDF5 file. Available keys: {list(h5_file.keys())}")
            if self.labels_key is None:
                raise ValueError(f"Could not find label dataset in HDF5 file. Available keys: {list(h5_file.keys())}")
            
            print(f"Using '{self.images_key}' as image dataset and '{self.labels_key}' as label dataset")
            
            # Get number of samples
            self.num_samples = h5_file[self.images_key].shape[0]
            logger.info(f"Found {self.num_samples} samples in HDF5 file")
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Open file in read mode (not keeping it open to avoid issues with multiple workers)
        with h5py.File(self.h5_path, 'r') as h5_file:
            # Get image and label using discovered keys
            image = np.array(h5_file[self.images_key][idx])
            label = np.array(h5_file[self.labels_key][idx])
            
            # Convert label to integer
            if isinstance(label, np.ndarray) and label.size == 1:
                label = label.item()
            label = int(label)
        
        # Convert image to float tensor
        image = torch.from_numpy(image).float()
        
        # Apply transform if available
        if self.transform:
            image = self.transform(image)
            
        return image, label


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler
) -> Dict[str, float]:
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward and optimize with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model on the validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / total
    val_acc = correct / total
    
    return {"loss": val_loss, "accuracy": val_acc}


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""
    
    def __init__(
        self, 
        patience: int = 7, 
        min_delta: float = 0.0, 
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            mode: One of 'min' or 'max' to determine whether to look for min or max of monitored quantity
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, current_score: float) -> bool:
        """
        Returns True if early stopping condition is met
        
        Args:
            current_score: Current value of the monitored metric
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == 'min':
            # Lower is better
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            # Higher is better
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            self.early_stop = True
            
        return self.early_stop


def get_attribute(args: Union[argparse.Namespace, SimpleNamespace], name: str, default_value: Any = None) -> Any:
    """
    Safely get an attribute from args with fallback to default value
    
    This helper function handles different config formats and naming conventions.
    
    Args:
        args: The arguments namespace
        name: The attribute name to look for
        default_value: Default value if attribute is not found
    
    Returns:
        The attribute value or default value
    """
    # Try direct attribute access
    if hasattr(args, name):
        return getattr(args, name)
    
    # Try alternative naming conventions
    alt_names = []
    if '_' in name:
        # Convert snake_case to dash-case
        alt_names.append(name.replace('_', '-'))
    else:
        # Convert dash-case to snake_case
        alt_names.append(name.replace('-', '_'))
    
    # Try alternatives
    for alt_name in alt_names:
        if hasattr(args, alt_name):
            return getattr(args, alt_name)
    
    # Return default value if not found
    return default_value


def train_model(args: Union[argparse.Namespace, SimpleNamespace]):
    """Main training function"""
    # Set random seeds for reproducibility
    seed = get_attribute(args, 'seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get dataset parameters
    data_path = get_attribute(args, 'data_path', '/honey/nmep/medium-imagenet-96.hdf5')
    val_split = get_attribute(args, 'val_split', 0.1)
    num_workers = get_attribute(args, 'num_workers', 4)
    
    # Create dataset and dataloaders
    logger.info(f"Loading dataset from {data_path}")
    dataset = HDF5Dataset(data_path)
    
    # Use the discovered labels key for getting class count
    labels_key = dataset.labels_key
    
    # Split dataset into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = get_attribute(args, 'batch_size', 64)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Get model parameters
    model_type = get_attribute(args, 'model_type', '121')
    attention = get_attribute(args, 'attention', 'cbam')
    activation = get_attribute(args, 'activation', 'mish')
    attention_pooling = get_attribute(args, 'attention_pooling', False)
    stochastic_depth = get_attribute(args, 'stochastic_depth', 0.1)
    
    # Create model
    logger.info(f"Creating DenseNet-{model_type} model")
    
    # Get class count from the dataset
    with h5py.File(data_path, 'r') as h5_file:
        # Get unique labels - properly casting to numpy array first
        labels = np.array(h5_file[labels_key])
        num_classes = len(np.unique(labels))
    
    logger.info(f"Dataset has {num_classes} classes")
    
    # initialized model based on model type
    if model_type == '121':
        model = DenseNet121(
            num_classes=num_classes,
            small_inputs=True,
            use_attention=attention,
            activation=activation,
            use_attention_pooling=attention_pooling,
            stochastic_depth_prob=stochastic_depth
        )
    elif model_type == '169':
        model = DenseNet169(
            num_classes=num_classes,
            small_inputs=True,
            use_attention=attention,
            activation=activation,
            use_attention_pooling=attention_pooling,
            stochastic_depth_prob=stochastic_depth
        )
    elif model_type == '201':
        model = DenseNet201(
            num_classes=num_classes,
            small_inputs=True,
            use_attention=attention,
            activation=activation,
            use_attention_pooling=attention_pooling,
            stochastic_depth_prob=stochastic_depth
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    model = model.to(device)
    
    # Get training parameters
    learning_rate = get_attribute(args, 'learning_rate', 0.001)
    min_lr = get_attribute(args, 'min_lr', 1e-6)
    weight_decay = get_attribute(args, 'weight_decay', 1e-4)
    scheduler_type = get_attribute(args, 'scheduler', 'cosine')
    early_stopping_patience = get_attribute(args, 'early_stopping_patience', 10)
    epochs = get_attribute(args, 'epochs', 100)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Set up learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr
        )
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=min_lr
        )
    else:
        scheduler = None
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        mode='min'
    )
    
    # Set up gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Get other parameters
    checkpoint_dir = get_attribute(args, 'checkpoint_dir', 'checkpoints')
    checkpoint_freq = get_attribute(args, 'checkpoint_freq', 5)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting training")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Update learning rate
        if scheduler is not None:
            if scheduler_type == 'plateau':
                # ReduceLROnPlateau scheduler takes a metrics value
                assert isinstance(scheduler, ReduceLROnPlateau)
                scheduler.step(val_metrics['loss'])
            else:
                # CosineAnnealingLR scheduler doesn't need a metrics value
                assert isinstance(scheduler, CosineAnnealingLR)
                scheduler.step()
        
        # Print metrics
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Time: {epoch_time:.2f}s, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Save checkpoint if best validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'densenet_{model_type}_best.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
            }, checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'densenet_{model_type}_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Check early stopping
        if early_stopping(val_metrics['loss']):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Training complete!")
    return model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train DenseNet on medium-imagenet")
    
    # Dataset parameters
    parser.add_argument('--data-path', type=str, default='/honey/nmep/medium-imagenet-96.hdf5',
                        help='Path to the HDF5 dataset file')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Proportion of data to use for validation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='121', choices=['121', '169', '201'],
                        help='DenseNet model type (121, 169, or 201)')
    parser.add_argument('--attention', type=str, default='cbam', choices=['se', 'cbam', 'none'],
                        help='Type of attention mechanism to use')
    parser.add_argument('--activation', type=str, default='mish', choices=['swish', 'mish', 'relu'],
                        help='Activation function to use')
    parser.add_argument('--attention-pooling', action='store_true',
                        help='Use attention pooling instead of average pooling')
    parser.add_argument('--stochastic-depth', type=float, default=0.1,
                        help='Stochastic depth probability (0 to disable)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of total epochs to run')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Patience for early stopping')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args) 