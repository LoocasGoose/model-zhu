"""ResNet implementation taken from kuangliu on github
link: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
    Squeeze-and-Excitation Networks. arXiv:1709.01507
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.checkpoint  # Import for gradient checkpointing
import numpy as np
import math


class StochasticDepth(nn.Module):
    """
    Implements Stochastic Depth (https://arxiv.org/abs/1603.09382) for residual networks.
    During training, randomly drops residual branches with probability 1-survival_prob.
    During inference, all branches are kept but scaled by survival_prob.
    """
    def __init__(self, survival_prob=0.8):
        super(StochasticDepth, self).__init__()
        self.survival_prob = survival_prob
        
    def forward(self, x, residual):
        # During inference, scale the residual
        if not self.training:
            return x + self.survival_prob * residual
            
        # During training, randomly drop the residual
        shape = [x.shape[0]] + [1] * (x.ndim - 1)  # Shape for broadcasting
        random_tensor = torch.rand(shape, device=x.device) < self.survival_prob
        return x + random_tensor * residual


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channel, reduction=16):  # Changed reduction back to standard 16
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),  # Removed inplace operation
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Efficient pooling
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Apply scaling WITHOUT inplace operation
        return x * y.expand_as(x)  # Non-inplace multiplication


class ResNetBlock(nn.Module):
    expansion = 1  # Basic block doesn't change the number of channels
    
    def __init__(self, in_channels, out_channels, stride=1, use_se=False, survival_prob=1.0):
        """
        Residual block for ResNet architecture.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution and shortcut
            use_se: Whether to use squeeze-excitation
            survival_prob: Probability for stochastic depth (1.0 = keep all)
        """
        super(ResNetBlock, self).__init__()
        
        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # Non-inplace ReLU
        
        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation block (optional)
        self.se = SELayer(out_channels) if use_se else nn.Identity()
        
        # Shortcut connection
        if stride != 1 or in_channels != self.expansion * out_channels:
            # If dimensions change, use a shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        else:
            # Identity shortcut
            self.shortcut = nn.Identity()
        
        # Stochastic depth for improved regularization
        self.stochastic_depth = None
        if survival_prob < 1.0:
            self.stochastic_depth = StochasticDepth(survival_prob)
        
        # For gradient checkpointing - disabled by default
        self.use_checkpoint = False
    
    def _forward_impl(self, x):
        # Save input for the shortcut
        identity = self.shortcut(x)
        
        # First conv + bn + activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv + bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE if enabled
        out = self.se(out)
        
        # Apply stochastic depth if enabled
        if self.stochastic_depth is not None:
            out = self.stochastic_depth(out, identity)
        else:
            # Add shortcut to the output
            out = out + identity
        
        # Apply final activation (non-inplace)
        out = self.relu(out)
        
        return out

    def forward(self, x):
        """
        Compute a forward pass with optional gradient checkpointing to save memory.
        """
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)


class Bottleneck(nn.Module):
    expansion = 4  # Bottleneck block expands the number of channels by 4
    
    def __init__(self, in_channels, out_channels, stride=1, use_se=False, survival_prob=1.0):
        """
        Bottleneck block for ResNet50/101/152.
        """
        super(Bottleneck, self).__init__()
        
        # First 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv with reduced channels
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # Squeeze-and-Excitation block
        self.se = SELayer(out_channels * self.expansion) if use_se else nn.Identity()
        
        self.relu = nn.ReLU(inplace=False)  # Non-inplace ReLU
        
        # Shortcut connection
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Stochastic depth for improved regularization
        self.stochastic_depth = None
        if survival_prob < 1.0:
            self.stochastic_depth = StochasticDepth(survival_prob)
        
        # For gradient checkpointing - disabled by default
        self.use_checkpoint = False
    
    def _forward_impl(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply SE if enabled
        out = self.se(out)
        
        # Apply stochastic depth if enabled
        if self.stochastic_depth is not None:
            out = self.stochastic_depth(out, identity)
        else:
            # Add shortcut to the output
            out = out + identity
        
        # Apply final activation
        out = self.relu(out)
        
        return out

    def forward(self, x):
        """
        Forward pass with optional gradient checkpointing.
        """
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=200, zero_init_residual=False, use_se=False, dropout_rate=0, stochastic_depth_rate=0):
        """
        Initialize ResNet with the specified block type and layer configuration.
        
        Args:
            block: Block class (ResNetBlock or Bottleneck)
            layers: List of block counts for each layer
            num_classes: Number of output classes
            zero_init_residual: Whether to initialize the residual branch BN weight as zero
            use_se: Whether to use Squeeze-and-Excitation blocks
            dropout_rate: Dropout probability before the final fully connected layer
            stochastic_depth_rate: Rate of stochastic depth (0 means no stochastic depth)
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.use_se = use_se
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)  # Changed to non-inplace ReLU
        
        # Build network with stochastic depth if requested
        self.stochastic_depth_rate = stochastic_depth_rate
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, stochastic_depth_rate=stochastic_depth_rate, block_id=0, total_blocks=sum(layers))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, stochastic_depth_rate=stochastic_depth_rate, block_id=layers[0], total_blocks=sum(layers))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, stochastic_depth_rate=stochastic_depth_rate, block_id=layers[0]+layers[1], total_blocks=sum(layers))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, stochastic_depth_rate=stochastic_depth_rate, block_id=layers[0]+layers[1]+layers[2], total_blocks=sum(layers))
        
        # Pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNetBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1, stochastic_depth_rate=0, block_id=0, total_blocks=0):
        layers = []
        
        # Calculate stochastic depth probability for each block
        sd_probs = []
        if stochastic_depth_rate > 0 and total_blocks > 0:
            # Linear decay of survival probability from 1.0 to 1-stochastic_depth_rate
            # Later blocks have higher dropout probability (lower survival rate)
            for i in range(blocks):
                # Calculate the current block position in the overall architecture
                curr_block_position = block_id + i
                # Linearly decreasing survival probability
                sd_prob = 1.0 - (curr_block_position / total_blocks) * stochastic_depth_rate
                sd_probs.append(sd_prob)
        else:
            # If stochastic depth is disabled, use 1.0 for all blocks
            sd_probs = [1.0] * blocks
        
        # First block with possibly downsampling (stride > 1)
        layers.append(block(self.in_channels, out_channels, stride, self.use_se, survival_prob=sd_probs[0]))
        
        # Update in_channels after applying first block
        self.in_channels = out_channels * block.expansion
        
        # Rest of the blocks with stride=1
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_se=self.use_se, survival_prob=sd_probs[i]))
        
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def forward(self, x):
        return self._forward_impl(x)
    
    def get_features(self, x, layer=4):
        """
        Extract features from a specific layer for transfer learning or visualization
        
        Args:
            x: Input tensor
            layer: Layer number to extract features from (1-4)
            
        Returns:
            Features from the specified layer
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if layer == 0:
            return x
        
        x = self.layer1(x)
        if layer == 1:
            return x
        
        x = self.layer2(x)
        if layer == 2:
            return x
        
        x = self.layer3(x)
        if layer == 3:
            return x
        
        x = self.layer4(x)
        if layer == 4:
            return x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# Learning rate scheduling functions
def cosine_annealing_lr(epoch, total_epochs, initial_lr, min_lr=0):
    """
    Cosine annealing learning rate schedule
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Current learning rate
    """
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))


def step_lr(epoch, step_size, gamma, initial_lr):
    """
    Step learning rate schedule
    
    Args:
        epoch: Current epoch
        step_size: Epochs per step
        gamma: Learning rate decay factor
        initial_lr: Initial learning rate
        
    Returns:
        Current learning rate
    """
    return initial_lr * (gamma ** (epoch // step_size))


def warmup_cosine_annealing_lr(epoch, total_epochs, warmup_epochs, initial_lr, min_lr=0):
    """
    Cosine annealing learning rate schedule with warmup
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Current learning rate
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing after warmup
        return min_lr + 0.5 * (initial_lr - min_lr) * (
            1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        )


def one_cycle_lr(epoch, total_epochs, initial_lr, max_lr, min_lr=0, div_factor=25.):
    """
    One-cycle learning rate policy
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        initial_lr: Initial learning rate
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        div_factor: Division factor for initial and final learning rate
        
    Returns:
        Current learning rate
    """
    # Calculate the halfway point
    half_cycle = total_epochs // 2
    
    if epoch < half_cycle:
        # First half: LR increases from initial_lr to max_lr
        return initial_lr + (max_lr - initial_lr) * (epoch / half_cycle)
    else:
        # Second half: LR decreases from max_lr to min_lr
        return max_lr - (max_lr - min_lr) * ((epoch - half_cycle) / (total_epochs - half_cycle))


# Data augmentation techniques
def mixup_data(x, y, alpha=1.0):
    """
    Applies mixup augmentation to the batch.
    
    Args:
        x: Input batch
        y: Target batch
        alpha: Mixup alpha parameter
        
    Returns:
        Mixed input, target_a, target_b, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Applies CutMix augmentation to the batch.
    
    Args:
        x: Input batch
        y: Target batch
        alpha: CutMix alpha parameter
        
    Returns:
        Mixed input, target_a, target_b, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    W, H = x.size()[2], x.size()[3]
    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    # Determine center of cutout
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Boundary
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x_aug = x.clone()
    x_aug[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match actual area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x_aug, y, y[index], lam


def label_smoothing_loss(outputs, targets, smoothing=0.1):
    """
    Apply label smoothing to the loss function
    
    Args:
        outputs: Model predictions
        targets: Ground truth labels
        smoothing: Smoothing factor
        
    Returns:
        Smoothed loss
    """
    num_classes = outputs.size(1)
    targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
    targets_smoothed = targets_one_hot * (1 - smoothing) + smoothing / num_classes
    
    log_probs = F.log_softmax(outputs, dim=1)
    loss = -(targets_smoothed * log_probs).sum(dim=1).mean()
    return loss


# Model configurations
def resnet18(num_classes=200, **kwargs):
    """Constructs a ResNet-18 model."""
    return ResNet(ResNetBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def resnet34(num_classes=200, **kwargs):
    """Constructs a ResNet-34 model."""
    return ResNet(ResNetBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet50(num_classes=200, **kwargs):
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet101(num_classes=200, **kwargs):
    """Constructs a ResNet-101 model."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)

def resnet152(num_classes=200, **kwargs):
    """Constructs a ResNet-152 model."""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


# For backward compatibility, renaming ResNet18 to match the original class name
# Remove this when migrating fully to the more general implementation
ResNet18 = resnet18
