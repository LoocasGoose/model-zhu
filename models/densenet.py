"""
COMPETITION DESTROYER 3000 !!!!

DenseNet implementation for image classification
Based on the paper: "Densely Connected Convolutional Networks" by Huang et al.
https://arxiv.org/abs/1608.06993
"""

from typing import List, Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Bottleneck(nn.Module):
    """
    Bottleneck block for DenseNet: BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)
    """
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4) -> None:
        """
        Initialize a bottleneck block.
        
        Args:
            in_channels: Number of input channels
            growth_rate: Growth rate (k) - number of feature maps each layer produces
            bn_size: Bottleneck size - multiplier for 1x1 conv layer
        """
        super(Bottleneck, self).__init__()
        inter_channels = bn_size * growth_rate
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the bottleneck block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with new features
        """
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class Transition(nn.Module):
    """
    Transition layer between dense blocks to reduce feature map size: BN-Conv(1x1)-AvgPool(2x2)
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize a transition layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the transition layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Downsampled tensor
        """
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseBlock(nn.Module):
    """
    Dense block with multiple bottleneck layers and dense connections
    """
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int = 4) -> None:
        """
        Initialize a dense block.
        
        Args:
            num_layers: Number of bottleneck layers in the block
            in_channels: Number of input channels
            growth_rate: Growth rate (k) - number of feature maps each layer produces
            bn_size: Bottleneck size - multiplier for 1x1 conv layer
        """
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate, bn_size))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the dense block.
        
        Args:
            x: Input tensor
            
        Returns:
            Concatenated tensor with all feature maps
        """
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    """
    DenseNet architecture with dense connections
    """
    def __init__(
        self, 
        growth_rate: int = 32, 
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64, 
        bn_size: int = 4, 
        compression_rate: float = 0.5,
        num_classes: int = 1000,
        small_inputs: bool = True
    ) -> None:
        """
        Initialize DenseNet.
        
        Args:
            growth_rate: Growth rate (k) - number of feature maps each layer produces
            block_config: Number of layers in each dense block
            num_init_features: Number of feature maps after initial convolution
            bn_size: Bottleneck size - multiplier for 1x1 conv layer
            compression_rate: Compression factor for transition layers
            num_classes: Number of output classes
            small_inputs: If True, use smaller stem for small inputs like CIFAR
        """
        super(DenseNet, self).__init__()
        
        # First convolution layer
        if small_inputs:
            # For small inputs like CIFAR
            self.features = nn.Sequential(
                nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True)
            )
        else:
            # For larger inputs like ImageNet
            self.features = nn.Sequential(
                nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        # Dense blocks with transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Add a dense block
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # Add a transition layer except after the last block
            if i != len(block_config) - 1:
                out_features = int(num_features * compression_rate)
                trans = Transition(num_features, out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Global pooling and classifier
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Class logits
        """
        features = self.features(x)
        out = F.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def DenseNet121(num_classes: int = 1000, small_inputs: bool = True) -> DenseNet:
    """
    Constructs a DenseNet-121 model.
    
    Args:
        num_classes: Number of output classes
        small_inputs: If True, use smaller stem for small inputs like CIFAR
    
    Returns:
        DenseNet-121 model
    """
    return DenseNet(
        growth_rate=32, 
        block_config=(6, 12, 24, 16),
        num_init_features=64, 
        num_classes=num_classes,
        small_inputs=small_inputs
    )


def DenseNet169(num_classes: int = 1000, small_inputs: bool = True) -> DenseNet:
    """
    Constructs a DenseNet-169 model.
    
    Args:
        num_classes: Number of output classes
        small_inputs: If True, use smaller stem for small inputs like CIFAR
    
    Returns:
        DenseNet-169 model
    """
    return DenseNet(
        growth_rate=32, 
        block_config=(6, 12, 32, 32),
        num_init_features=64, 
        num_classes=num_classes,
        small_inputs=small_inputs
    )


def DenseNet201(num_classes: int = 1000, small_inputs: bool = True) -> DenseNet:
    """
    Constructs a DenseNet-201 model.
    
    Args:
        num_classes: Number of output classes
        small_inputs: If True, use smaller stem for small inputs like CIFAR
    
    Returns:
        DenseNet-201 model
    """
    return DenseNet(
        growth_rate=32, 
        block_config=(6, 12, 48, 32),
        num_init_features=64, 
        num_classes=num_classes,
        small_inputs=small_inputs
    )

