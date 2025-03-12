"""
Advanced data augmentation techniques for improved training.
"""

import numpy as np
import torch
import torch.nn as nn
import random
from typing import Optional, Tuple, Dict, Any, Union


def mixup_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 1.0, 
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies Mixup augmentation to the batch.
    Paper: https://arxiv.org/abs/1710.09412
    
    Args:
        x: Input tensor (batch of images)
        y: Target tensor (batch of labels)
        alpha: Mixup interpolation coefficient
        device: Device to use
        
    Returns:
        mixed_x: Mixed input
        y_a: First labels
        y_b: Second labels
        lam: Lambda coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module, 
    pred: torch.Tensor, 
    y_a: torch.Tensor, 
    y_b: torch.Tensor, 
    lam: float
) -> torch.Tensor:
    """
    Mixup criterion that applies the weighted loss.
    
    Args:
        criterion: Loss function
        pred: Predictions
        y_a: First labels
        y_b: Second labels
        lam: Lambda coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 1.0, 
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies CutMix augmentation to the batch.
    Paper: https://arxiv.org/abs/1905.04899
    
    Args:
        x: Input tensor (batch of images)
        y: Target tensor (batch of labels)
        alpha: CutMix interpolation coefficient
        device: Device to use
        
    Returns:
        mixed_x: Mixed input
        y_a: First labels
        y_b: Second labels
        lam: Lambda coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    # Get bbox dimensions
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # Apply cutmix
    x_cut = x.clone()
    x_cut[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to account for exact ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x_cut, y, y[index], lam


def rand_bbox(
    size: Tuple[int, ...], 
    lam: float
) -> Tuple[int, int, int, int]:
    """
    Generates a random bounding box for CutMix.
    
    Args:
        size: Input size (B, C, H, W)
        lam: Lambda coefficient
        
    Returns:
        Bounding box coordinates (x1, y1, x2, y2)
    """
    W = size[2]
    H = size[3]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class RandomErasing:
    """
    Random Erasing augmentation.
    Paper: https://arxiv.org/abs/1708.04896
    """
    def __init__(
        self, 
        probability: float = 0.5, 
        area_ratio_range: Tuple[float, float] = (0.02, 0.4),
        aspect_ratio_range: Tuple[float, float] = (0.3, 3.3),
        min_erased_area_pct: float = 0.02, 
        max_erased_area_pct: float = 0.4, 
        mode: str = 'const', 
        fill_value: Union[int, Tuple[int, ...]] = 0
    ):
        """
        Initialize Random Erasing.
        
        Args:
            probability: Probability of applying random erasing
            area_ratio_range: Range of area ratio to erase
            aspect_ratio_range: Range of aspect ratio to erase
            min_erased_area_pct: Minimum percentage of erased area
            max_erased_area_pct: Maximum percentage of erased area
            mode: Erasing mode ('const', 'rand', 'pixel')
            fill_value: Value to fill erased area (for 'const' mode)
        """
        self.probability = probability
        self.area_ratio_range = area_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.min_erased_area_pct = min_erased_area_pct
        self.max_erased_area_pct = max_erased_area_pct
        self.mode = mode
        self.fill_value = fill_value
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random erasing to the input image.
        
        Args:
            img: Input tensor of shape (C, H, W)
            
        Returns:
            Augmented image
        """
        if random.random() > self.probability:
            return img
        
        img_c, img_h, img_w = img.shape
        
        # Get random area
        area = img_h * img_w
        target_area = random.uniform(self.min_erased_area_pct, self.max_erased_area_pct) * area
        aspect_ratio = random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])
        
        # Calculate erasing size
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if w < img_w and h < img_h:
            # Random position
            i = random.randint(0, img_h - h)
            j = random.randint(0, img_w - w)
            
            if self.mode == 'const':
                if isinstance(self.fill_value, (int, float)):
                    img[:, i:i+h, j:j+w] = self.fill_value
                else:
                    for c in range(img_c):
                        fill_c = self.fill_value[c] if c < len(self.fill_value) else self.fill_value[0]
                        img[c, i:i+h, j:j+w] = fill_c
            elif self.mode == 'rand':
                img[:, i:i+h, j:j+w] = torch.empty_like(img[:, i:i+h, j:j+w]).normal_()
            elif self.mode == 'pixel':
                # Randomly sample pixels from the image
                idx = torch.randint(0, area, (h * w,))
                idx = torch.stack([idx // img_w, idx % img_w], dim=1)
                for c in range(img_c):
                    pixel_c = img[c][idx[:, 0], idx[:, 1]].reshape(h, w)
                    img[c, i:i+h, j:j+w] = pixel_c
                
        return img


def apply_augmentations(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    config: Dict[str, Any], 
    mode: str = 'train'
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[float]]:
    """
    Apply configured augmentations to a batch of data.
    
    Args:
        inputs: Input tensor (batch of images)
        targets: Target tensor (batch of labels)
        config: Configuration dictionary
        mode: 'train' or 'val' mode
        
    Returns:
        Augmented inputs, target_a, target_b (if mixup/cutmix), lambda (if mixup/cutmix)
    """
    device = inputs.device
    
    if mode != 'train':
        return inputs, targets, None, None
    
    # Apply CutMix with probability
    if getattr(config.AUG, 'USE_CUTMIX', False) and random.random() < 0.5:
        inputs, targets_a, targets_b, lam = cutmix_data(
            inputs, targets, 
            alpha=getattr(config.AUG, 'CUTMIX_ALPHA', 1.0),
            device=device
        )
        return inputs, targets_a, targets_b, lam
    
    # Apply Mixup with probability
    if getattr(config.AUG, 'USE_MIXUP', False) and random.random() < 0.5:
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, 
            alpha=getattr(config.AUG, 'MIXUP_ALPHA', 0.2),
            device=device
        )
        return inputs, targets_a, targets_b, lam
    
    # Apply Random Erasing
    if getattr(config.AUG, 'USE_RANDOM_ERASING', False):
        erasing = RandomErasing(
            probability=getattr(config.AUG, 'RANDOM_ERASING_PROB', 0.25),
            mode='const',
            fill_value=0
        )
        for i in range(inputs.size(0)):
            inputs[i] = erasing(inputs[i])
    
    return inputs, targets, None, None


def mixup_criterion_wrapper(
    criterion: nn.Module, 
    outputs: torch.Tensor, 
    targets_a: torch.Tensor, 
    targets_b: Optional[torch.Tensor] = None, 
    lam: Optional[float] = None
) -> torch.Tensor:
    """
    Wrapper for criterion to handle both regular and mixup/cutmix loss.
    
    Args:
        criterion: Loss function
        outputs: Predictions
        targets_a: First targets
        targets_b: Second targets (for mixup/cutmix)
        lam: Lambda value (for mixup/cutmix)
        
    Returns:
        Loss value
    """
    if targets_b is not None and lam is not None:
        return mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
    else:
        return criterion(outputs, targets_a) 