"""ResNeXt implementation based on the ResNet architecture
For Pre-activation ResNeXt, see 'preact_resnext.py'.
Reference:
[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
    Aggregated Residual Transformations for Deep Neural Networks. arXiv:1611.05431
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint
from torch.nn import Dropout2d


class ResNeXtBlock(nn.Module):
    """
    A block for ResNeXt architecture that uses grouped convolutions for the 
    split-transform-merge strategy with cardinality.
    """
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, pruning_rate=1.0, activation='relu', drop_rate=0.0):
        """
        Create a ResNeXt block with cardinality.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first conv layer
            cardinality: Number of transformation groups (cardinality dimension)
            base_width: Base width per group, controlling the bottleneck width
            pruning_rate: Channel pruning rate (1.0 means no pruning)
            activation: Activation function to use ('relu', 'relu6', or 'silu')
            drop_rate: Dropout rate
        """
        super(ResNeXtBlock, self).__init__()
        
        # Calculate bottleneck width using standard reduction approach (similar to ResNet)
        # Standard bottleneck reduction factor
        reduction = 4
        # Calculate width based on input channels with bottleneck reduction
        bottleneck_width = int((in_channels // reduction) * cardinality * base_width / cardinality)
        bottleneck_width = int(bottleneck_width * pruning_rate)
        # Ensure width is a multiple of cardinality for efficient group distribution
        bottleneck_width = max((bottleneck_width // cardinality) * cardinality, cardinality)
        
        # First, bottleneck down to bottleneck_width channels using 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, bottleneck_width, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_width)
        
        # Split-transform using grouped convolutions
        # Use stride here for efficiency (replaces max pooling with strided conv)
        self.conv2 = nn.Conv2d(bottleneck_width, bottleneck_width, kernel_size=3, stride=stride, padding=1, 
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_width)
        
        # Bottleneck up to out_channels using 1x1 conv
        self.conv3 = nn.Conv2d(bottleneck_width, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Select activation function based on parameter
        if activation == 'relu6':
            self.relu = nn.ReLU6(inplace=True)
        elif activation == 'silu':
            self.relu = nn.SiLU(inplace=True)
        else:  # default to relu
            self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Add dropout layer
        self.dropout = Dropout2d(p=drop_rate) if drop_rate > 0 else nn.Identity()
        
        # Store cardinality for scaling in forward pass
        self.cardinality = cardinality

    def forward(self, x):
        """
        Forward pass through the ResNeXt block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through the block
        """
        identity = self.shortcut(x)
        
        # Bottleneck down
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Split-transform with grouped convolutions
        out = self.conv2(out)
        out = self.bn2(out)
        # No ReLU here - removed to improve gradient flow per standard ResNet/ResNeXt architecture
        
        # Bottleneck up
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply dropout BEFORE residual addition
        out = self.dropout(out)
        
        # Add residual connection and apply ReLU
        out += identity
        out = self.relu(out)
        
        return out


class ResNeXt(nn.Module):
    """
    ResNeXt model with configurable depth, based on the original paper.
    """
    def __init__(self, block, num_blocks, cardinality=32, base_width=4, pruning_rate=1.0, 
                 num_classes=200, activation='relu', use_checkpoint=False, drop_rate=0.0, small_input=False):
        """
        Initialize ResNeXt model.
        
        Args:
            block: Block class to use (ResNeXtBlock)
            num_blocks: List containing number of blocks in each layer
            cardinality: Number of transformation groups
            base_width: Base width per group
            pruning_rate: Channel pruning rate (1.0 means no pruning)
            num_classes: Number of output classes
            activation: Activation function to use ('relu', 'relu6', or 'silu')
            use_checkpoint: Whether to use checkpoint for memory efficiency
            drop_rate: Feature dropout rate within blocks
            small_input: Whether using small input images (e.g., CIFAR)
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.pruning_rate = pruning_rate
        self.in_channels = 64
        self.activation = activation
        self.use_checkpoint = use_checkpoint
        self.drop_rate = drop_rate
        self.small_input = small_input
        
        # Initial convolution layer - different for small inputs vs ImageNet-scale
        if small_input:
            # Small input (e.g., CIFAR) - use 3x3 conv without pooling
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.Identity()  # No pooling for small inputs
        else:
            # ImageNet-scale input - use 7x7 conv with pooling (original paper)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Select activation function based on parameter
        if activation == 'relu6':
            self.relu = nn.ReLU6(inplace=True)
        elif activation == 'silu':
            self.relu = nn.SiLU(inplace=True)
        else:  # default to relu
            self.relu = nn.ReLU(inplace=True)
        
        # ResNeXt layers with increasing feature map dimensions
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Create a layer of blocks.
        
        Args:
            block: Block class to use
            out_channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for the first block
            
        Returns:
            Sequential container with blocks
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(
                block(
                    self.in_channels, 
                    out_channels, 
                    stride, 
                    self.cardinality, 
                    self.base_width,
                    self.pruning_rate,
                    self.activation,
                    self.drop_rate  # Use the model's drop_rate instead of hardcoding 0.0
                )
            )
            self.in_channels = out_channels
            
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        Initialize model weights for better training convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x):
        """
        Implementation of the forward pass.
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # Apply max pooling if not small_input
        
        # Handle checkpointing if enabled during training
        if self.use_checkpoint and self.training:
            # Only apply checkpointing to deeper layers during training for memory efficiency
            x = self.layer1(x)  # First layer is shallow, no checkpoint needed
            x = checkpoint(self.layer2, x)
            x = checkpoint(self.layer3, x)
            x = checkpoint(self.layer4, x)
        else:
            # Standard forward pass through all layers (more reliable for accuracy)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        # Pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        
        return x

    def forward(self, x):
        """
        Forward pass through the ResNeXt model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with class predictions
        """
        return self._forward_impl(x)


# Factory functions for different ResNeXt configurations

def ResNeXt50(num_classes=200, cardinality=32, base_width=4, pruning_rate=1.0, activation='relu', use_checkpoint=False, drop_rate=0.0, small_input=False):
    """
    ResNeXt-50 model with 32x4d configuration (32 groups, 4 channels per group).
    
    Args:
        num_classes: Number of output classes
        cardinality: Number of transformation groups
        base_width: Base width per group
        pruning_rate: Channel pruning rate (1.0 means no pruning)
        activation: Activation function to use
        use_checkpoint: Whether to use checkpoint for memory efficiency
        drop_rate: Feature dropout rate within blocks
        small_input: Whether input images are small (CIFAR) or large (ImageNet)
        
    Returns:
        ResNeXt-50 model
    """
    return ResNeXt(ResNeXtBlock, [3, 4, 6, 3], cardinality, base_width, 
                   pruning_rate, num_classes, activation, use_checkpoint, 
                   drop_rate, small_input)


def ResNeXt101(num_classes=200, cardinality=32, base_width=4, pruning_rate=1.0, activation='relu', use_checkpoint=False, drop_rate=0.0, small_input=False):
    """
    ResNeXt-101 model with 32x4d configuration (32 groups, 4 channels per group).
    
    Args:
        num_classes: Number of output classes
        cardinality: Number of transformation groups
        base_width: Base width per group
        pruning_rate: Channel pruning rate (1.0 means no pruning)
        activation: Activation function to use
        use_checkpoint: Whether to use checkpoint for memory efficiency
        drop_rate: Feature dropout rate within blocks
        small_input: Whether input images are small (CIFAR) or large (ImageNet)
        
    Returns:
        ResNeXt-101 model
    """
    return ResNeXt(ResNeXtBlock, [3, 4, 23, 3], cardinality, base_width, 
                   pruning_rate, num_classes, activation, use_checkpoint, 
                   drop_rate, small_input)


def ResNeXt29(num_classes=200, cardinality=16, base_width=64, pruning_rate=1.0, activation='relu', use_checkpoint=False, drop_rate=0.0, small_input=True):
    """
    ResNeXt-29 model optimized for CIFAR, with 16x64d configuration.
    
    Args:
        num_classes: Number of output classes
        cardinality: Number of transformation groups
        base_width: Base width per group
        pruning_rate: Channel pruning rate (1.0 means no pruning)
        activation: Activation function to use
        use_checkpoint: Whether to use checkpoint for memory efficiency
        drop_rate: Feature dropout rate within blocks
        small_input: Whether input images are small (CIFAR) or large (ImageNet)
        
    Returns:
        ResNeXt-29 model
    """
    return ResNeXt(ResNeXtBlock, [3, 3, 3, 3], cardinality, base_width, 
                   pruning_rate, num_classes, activation, use_checkpoint, 
                   drop_rate, small_input)
