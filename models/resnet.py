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
import math


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def __init__(self, inplace=False):
        super(Mish, self).__init__()
        self.inplace = inplace  # Inplace flag is kept for API consistency but not used

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            Mish(),  # Replace ReLU with Mish
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class StochasticDepth(nn.Module):
    """
    Implements Stochastic Depth as described in https://arxiv.org/abs/1603.09382
    Randomly drops the residual connection with probability 1-survival_prob during training
    """
    def __init__(self, survival_prob=0.8):
        super(StochasticDepth, self).__init__()
        self.survival_prob = survival_prob
        
    def forward(self, x, residual):
        if not self.training:
            return x + residual
            
        # During training, randomly drop the residual with probability 1-survival_prob
        random_tensor = torch.rand([x.shape[0], 1, 1, 1], device=x.device) < self.survival_prob
        return x + random_tensor * residual


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True, survival_prob=1.0):
        """
        Create a memory-optimized residual block for our ResNet18 architecture.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution and shortcut
            use_se: Whether to use squeeze-excitation
            survival_prob: Probability of keeping the residual connection (stochastic depth)
        """
        super(ResNetBlock, self).__init__()
        
        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation block
        self.use_se = use_se
        if use_se:
            self.se = SEModule(out_channels)
        
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
            
        # Use Mish activation to improve performance
        self.activation = Mish()
        
        # Stochastic depth for regularization
        self.stochastic_depth = StochasticDepth(survival_prob) if survival_prob < 1.0 else None

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
        out = self.activation(out)
        
        # Second conv + bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE attention if enabled
        if self.use_se:
            out = self.se(out)
        
        # Add shortcut to the output with stochastic depth if enabled
        if self.stochastic_depth is not None:
            out = self.stochastic_depth(out, identity)
        else:
            out = out + identity
            
        out = self.activation(out)
        
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=200, enable_checkpoint=False, use_se=True, stochastic_depth_rate=0.1, dropout_rate=0.2):
        num_classes = num_classes
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # Replace 3x3 conv with standard 7x7 conv with stride 2 (from original paper)
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = Mish()  # Use Mish activation instead of ReLU
        # Add maxpool layer after initial conv (original paper design)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Store flags
        self.use_se = use_se
        self.stochastic_depth_rate = stochastic_depth_rate
        
        # Create layers with stochastic depth (higher survival probability for earlier layers)
        num_blocks = [2, 2, 2, 2]  # ResNet18 has 2 blocks per layer
        total_blocks = sum(num_blocks)
        self.layer1 = self.make_block(out_channels=64, stride=1, num_blocks=num_blocks[0], block_id=0, total_blocks=total_blocks)
        self.layer2 = self.make_block(out_channels=128, stride=2, num_blocks=num_blocks[1], block_id=num_blocks[0], total_blocks=total_blocks)
        self.layer3 = self.make_block(out_channels=256, stride=2, num_blocks=num_blocks[2], block_id=sum(num_blocks[:2]), total_blocks=total_blocks)
        self.layer4 = self.make_block(out_channels=512, stride=2, num_blocks=num_blocks[3], block_id=sum(num_blocks[:3]), total_blocks=total_blocks)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Add dropout before final classifier for better generalization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.linear = nn.Linear(512, num_classes)
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Adjust initialization for Mish activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Proper linear layer initialization
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
        # Store gradient checkpointing flag
        self.enable_checkpoint = enable_checkpoint

    def make_block(self, out_channels, stride, num_blocks, block_id, total_blocks):
        layers = []
        
        # Apply stochastic depth with linearly decreasing survival probability
        for i, cur_stride in enumerate([stride, 1]):
            # Calculate the survival probability
            # Earlier blocks have higher survival probability
            if self.stochastic_depth_rate > 0:
                cur_block_id = block_id + i
                survival_prob = 1.0 - self.stochastic_depth_rate * float(cur_block_id) / total_blocks
                survival_prob = max(survival_prob, 0.5)  # Don't go below 0.5
            else:
                survival_prob = 1.0
                
            layers.append(ResNetBlock(
                self.in_channels, 
                out_channels, 
                cur_stride, 
                use_se=self.use_se,
                survival_prob=survival_prob
            ))
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
        x = self.activation(x)
        x = self.maxpool(x)
        
        # Apply layers with optional gradient checkpointing
        x = self._run_layer(self.layer1, x)
        x = self._run_layer(self.layer2, x)
        x = self._run_layer(self.layer3, x)
        x = self._run_layer(self.layer4, x)
        
        # Final pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout before final classifier 
        x = self.linear(x)
        return x
