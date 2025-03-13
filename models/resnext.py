"""ResNeXt implementation based on the ResNet architecture
For Pre-activation ResNeXt, see 'preact_resnext.py'.
Reference:
[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
    Aggregated Residual Transformations for Deep Neural Networks. arXiv:1611.05431
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class ResNeXtBlock(nn.Module):
    """
    A block for ResNeXt architecture that uses grouped convolutions for the 
    split-transform-merge strategy with cardinality.
    """
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4):
        """
        Create a ResNeXt block with cardinality.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first conv layer
            cardinality: Number of transformation groups (cardinality dimension)
            base_width: Base width for each group, controlling the bottleneck width
        """
        super(ResNeXtBlock, self).__init__()
        
        # Calculate width for each group using cardinality and base_width
        width = cardinality * base_width
        
        # First, bottleneck down to 'width' channels using 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        # Split-transform using grouped convolutions
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, 
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        # Bottleneck up to out_channels using 1x1 conv
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

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
        out = self.relu(out)
        
        # Bottleneck up
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Add residual connection and apply ReLU
        out += identity
        out = self.relu(out)
        
        return out


class ResNeXt(nn.Module):
    """
    ResNeXt model with configurable depth, based on the original paper.
    """
    def __init__(self, block, num_blocks, cardinality=32, base_width=4, num_classes=200):
        """
        Initialize ResNeXt model.
        
        Args:
            block: Block class to use (ResNeXtBlock)
            num_blocks: List containing number of blocks in each layer
            cardinality: Number of transformation groups
            base_width: Base width per group
            num_classes: Number of output classes
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNeXt layers with increasing feature map dimensions
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

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
                    self.base_width
                )
            )
            self.in_channels = out_channels
            
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the ResNeXt model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with class predictions
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # ResNeXt layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        
        return x


# Factory functions for different ResNeXt configurations

def ResNeXt50(num_classes=200, cardinality=32, base_width=4):
    """
    ResNeXt-50 model with 32x4d configuration (32 groups, 4 channels per group).
    
    Args:
        num_classes: Number of output classes
        cardinality: Number of transformation groups
        base_width: Base width per group
        
    Returns:
        ResNeXt-50 model
    """
    return ResNeXt(ResNeXtBlock, [3, 4, 6, 3], cardinality, base_width, num_classes)


def ResNeXt101(num_classes=200, cardinality=32, base_width=4):
    """
    ResNeXt-101 model with 32x4d configuration (32 groups, 4 channels per group).
    
    Args:
        num_classes: Number of output classes
        cardinality: Number of transformation groups
        base_width: Base width per group
        
    Returns:
        ResNeXt-101 model
    """
    return ResNeXt(ResNeXtBlock, [3, 4, 23, 3], cardinality, base_width, num_classes)


def ResNeXt29(num_classes=200, cardinality=16, base_width=64):
    """
    ResNeXt-29 model optimized for CIFAR, with 16x64d configuration.
    
    Args:
        num_classes: Number of output classes
        cardinality: Number of transformation groups
        base_width: Base width per group
        
    Returns:
        ResNeXt-29 model
    """
    return ResNeXt(ResNeXtBlock, [3, 3, 3, 3], cardinality, base_width, num_classes)
