"""ResNet implementation taken from kuangliu on github
link: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.checkpoint  # For gradient checkpointing


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Create a memory-optimized residual block for our ResNet18 architecture.

        Here is the expected network structure:
        - conv layer with
            out_channels=out_channels, 3x3 kernel, stride=stride
        - batchnorm layer (Batchnorm2D)
        - conv layer with
            out_channels=out_channels, 3x3 kernel, stride=1
        - batchnorm layer (Batchnorm2D)
        - shortcut layer:
            if either the stride is not 1 or the out_channels is not equal to in_channels:
                the shortcut layer is composed of two steps:
                - conv layer with
                    in_channels=in_channels, out_channels=out_channels, 1x1 kernel, stride=stride
                - batchnorm layer (Batchnorm2D)
            else:
                the shortcut layer should be an no-op
        """
        super(ResNetBlock, self).__init__()
        
        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            # If dimensions change, shortcut needs to adjust dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Identity shortcut - more efficient than empty Sequential
            self.shortcut = nn.Identity()
            
        # Use inplace ReLU to save memory
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Compute a forward pass.
        
        x: batch of images of shape (batch_size, num_channels, width, height)
        returns: result of passing x through this block
        """
        # Save input for the shortcut
        identity = self.shortcut(x)
        
        # First conv + bn + activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv + bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add shortcut to the output and apply activation
        out += identity  # Using += is more memory efficient
        out = self.relu(out)
        
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=200, enable_checkpoint=False):
        num_classes = num_classes
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_block(out_channels=64, stride=1)
        self.layer2 = self.make_block(out_channels=128, stride=2)
        self.layer3 = self.make_block(out_channels=256, stride=2)
        self.layer4 = self.make_block(out_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Store gradient checkpointing flag
        self.enable_checkpoint = enable_checkpoint

    def make_block(self, out_channels, stride):
        # Read the following, and uncomment it when you understand it, no need to add more code
        layers = []
        for stride in [stride, 1]:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
        
    def _run_layer(self, layer, x):
        # Helper function to run a layer with potential checkpointing
        if self.enable_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(layer, x, preserve_rng_state=False)
        else:
            return layer(x)

    def forward(self, x):
        # Initial convolution and batch norm
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Apply layers with optional gradient checkpointing
        x = self._run_layer(self.layer1, x)
        x = self._run_layer(self.layer2, x)
        x = self._run_layer(self.layer3, x)
        x = self._run_layer(self.layer4, x)
        
        # Final pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
