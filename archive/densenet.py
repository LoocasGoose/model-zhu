"""
DenseNet implementation with attention mechanisms and advanced activations
Based on the paper: "Densely Connected Convolutional Networks" by Huang et al.
with additional optimizations for improved accuracy.
"""
from typing import List, Tuple, Optional, Union, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwishActivation(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    Paper: https://arxiv.org/abs/1710.05941
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.beta * x)


class MishActivation(nn.Module):
    """
    Mish activation function: x * tanh(softplus(x))
    Paper: https://arxiv.org/abs/1908.08681
    Often provides better performance than ReLU or Swish
    """
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))


def get_activation(name: str = "swish") -> nn.Module:
    """
    Return activation function based on name.
    
    Args:
        name: Name of the activation function ('swish', 'mish', 'relu')
        
    Returns:
        Activation module
    """
    if name == "mish":
        return MishActivation()
    elif name == "relu":
        return nn.ReLU(inplace=True)
    else:  # default to swish
        return SwishActivation()


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    Paper: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels: int, reduction_ratio: int = 16, activation: Optional[nn.Module] = None):
        super().__init__()
        if activation is None:
            activation = SwishActivation()
            
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            activation,
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for focusing on important spatial locations
    Used as part of CBAM
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # Generate spatial attention map using average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention_map = self.sigmoid(attention)
        return x * attention_map


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Combines channel and spatial attention mechanisms
    Paper: https://arxiv.org/abs/1807.06521
    """
    def __init__(self, channels: int, reduction_ratio: int = 16, activation: Optional[nn.Module] = None):
        super().__init__()
        if activation is None:
            activation = SwishActivation()
            
        self.channel_attention = SEBlock(channels, reduction_ratio, activation)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        # Apply channel attention followed by spatial attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def get_attention_module(
    attention_type: str,
    channels: int, 
    reduction_ratio: int = 16,
    activation: Optional[nn.Module] = None
) -> Optional[nn.Module]:
    """
    Return appropriate attention module based on type.
    
    Args:
        attention_type: Type of attention ('se', 'cbam', 'none')
        channels: Number of input channels
        reduction_ratio: Reduction ratio for attention blocks
        activation: Activation function to use
    
    Returns:
        Attention module or None
    """
    if attention_type == "cbam":
        return CBAM(channels, reduction_ratio, activation)
    elif attention_type == "se":
        return SEBlock(channels, reduction_ratio, activation)
    else:
        return None


class AttentionPooling(nn.Module):
    """
    Global attention pooling layer.
    Computes spatial attention scores and uses them to aggregate features.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        # Project input features to a hidden dimension and then to 1-channel attention map
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        attn = F.relu(self.conv1(x))  # (B, hidden_dim, H, W)
        attn = self.conv2(attn)       # (B, 1, H, W)
        attn = attn.view(x.size(0), -1)  # Flatten spatial dims: (B, H*W)
        attn = F.softmax(attn, dim=1)    # Softmax over spatial locations
        attn = attn.view(x.size(0), 1, x.size(2), x.size(3))
        # Weighted sum of features
        pooled = (x * attn).sum(dim=(2,3))  # (B, C)
        return pooled


class HierarchicalAttentionPooling(nn.Module):
    """
    Hierarchical attention pooling combining multi-scale features
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.attention = CBAM(in_channels, reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x: Tensor) -> Tensor:
        # Apply attention mechanism
        x = self.attention(x)
        # Apply adaptive pooling
        x = self.pool(x)
        # Return flattened tensor for consistent interface
        return torch.flatten(x, 1)


class Bottleneck(nn.Module):
    """
    Bottleneck block for DenseNet with attention and advanced activations
    """
    def __init__(
        self, 
        in_channels: int, 
        growth_rate: int, 
        bn_size: int = 4, 
        use_attention: str = "se",  # Options: "se", "cbam", "none"
        reduction_ratio: int = 16, 
        dropout_rate: float = 0.0,
        activation: str = "swish"  # Options: "swish", "mish", "relu"
    ):
        """
        Initialize a bottleneck block.
        
        Args:
            in_channels: Number of input channels
            growth_rate: Growth rate (k) - number of feature maps each layer produces
            bn_size: Bottleneck size - multiplier for 1x1 conv layer
            use_attention: Type of attention mechanism to use
            reduction_ratio: Reduction ratio for attention blocks
            dropout_rate: Dropout rate for regularization
            activation: Type of activation function to use
        """
        super().__init__()
        inter_channels = bn_size * growth_rate
        
        # Get activation function
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
        # Get attention module
        self.attention = get_attention_module(use_attention, growth_rate, reduction_ratio, self.act1)
            
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the bottleneck block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with new features
        """
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        
        if self.attention is not None:
            out = self.attention(out)
            
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out


class Transition(nn.Module):
    """
    Transition layer between dense blocks
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        use_attention: str = "se",  # Options: "se", "cbam", "none" 
        reduction_ratio: int = 16, 
        dropout_rate: float = 0.0,
        activation: str = "swish"  # Options: "swish", "mish", "relu"
    ):
        """
        Initialize a transition layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_attention: Type of attention mechanism to use
            reduction_ratio: Reduction ratio for attention blocks
            dropout_rate: Dropout rate for regularization
            activation: Type of activation function to use
        """
        super().__init__()
        
        # Get activation function
        self.act = get_activation(activation)
            
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Get attention module
        self.attention = get_attention_module(use_attention, out_channels, reduction_ratio, self.act)
            
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the transition layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Downsampled tensor
        """
        out = self.conv(self.act(self.bn(x)))
        out = self.pool(out)
        
        if self.attention is not None:
            out = self.attention(out)
            
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out


class DenseBlock(nn.Module):
    """
    Dense block with attention mechanisms and stochastic depth
    """
    def __init__(
        self, 
        num_layers: int, 
        in_channels: int, 
        growth_rate: int, 
        bn_size: int = 4, 
        use_attention: str = "se",  # Options: "se", "cbam", "none"
        reduction_ratio: int = 16, 
        dropout_rate: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        activation: str = "swish"  # Options: "swish", "mish", "relu"
    ):
        """
        Initialize a dense block.
        
        Args:
            num_layers: Number of bottleneck layers in the block
            in_channels: Number of input channels
            growth_rate: Growth rate (k) - number of feature maps each layer produces
            bn_size: Bottleneck size - multiplier for 1x1 conv layer
            use_attention: Type of attention mechanism to use
            reduction_ratio: Reduction ratio for attention blocks
            dropout_rate: Dropout rate for regularization
            stochastic_depth_prob: Probability of dropping a layer in stochastic depth
            activation: Type of activation function to use
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.stochastic_depth_prob = stochastic_depth_prob
        
        for i in range(num_layers):
            self.layers.append(
                Bottleneck(
                    in_channels + i * growth_rate, 
                    growth_rate, 
                    bn_size, 
                    use_attention, 
                    reduction_ratio,
                    dropout_rate,
                    activation
                )
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the dense block with stochastic depth.
        
        Args:
            x: Input tensor
            
        Returns:
            Concatenated tensor with all feature maps
        """
        features = [x]
        
        for i, layer in enumerate(self.layers):
            # Calculate input features by concatenating all previous features
            input_features = torch.cat(features, 1)
            
            # Implement stochastic depth if training and probability > 0
            if self.training and self.stochastic_depth_prob > 0.0:
                # Calculate survival probability that decreases linearly with depth
                survival_prob = 1.0 - (i / self.num_layers) * self.stochastic_depth_prob
                
                if torch.rand(1).item() < survival_prob:
                    # Execute the layer and scale by survival probability for proper expectation
                    new_features = layer(input_features)
                    # Scale by 1/survival_prob to maintain expected value during training
                    new_features = new_features / survival_prob
                else:
                    # Skip the layer and create zero features
                    new_features = torch.zeros(
                        x.size(0), self.growth_rate, x.size(2), x.size(3),
                        device=x.device
                    )
            else:
                # Normal execution (no stochastic depth)
                new_features = layer(input_features)
                
            features.append(new_features)
            
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    """
    DenseNet architecture with attention mechanisms and improved regularization
    """
    def __init__(
        self, 
        growth_rate: int = 32, 
        block_config: Tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64, 
        bn_size: int = 4, 
        compression_rate: float = 0.5,
        num_classes: int = 1000,
        small_inputs: bool = True,
        use_attention: str = "se",  # Options: "se", "cbam", "none"
        reduction_ratio: int = 16,
        dropout_rate: float = 0.2,
        stochastic_depth_prob: float = 0.0,
        activation: str = "swish",  # Options: "swish", "mish", "relu"
        use_attention_pooling: bool = False
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
            use_attention: Type of attention mechanism to use
            reduction_ratio: Reduction ratio for attention blocks
            dropout_rate: Dropout rate for regularization
            stochastic_depth_prob: Probability of dropping a layer for stochastic depth
            activation: Type of activation function to use
            use_attention_pooling: Whether to use attention pooling instead of average pooling
        """
        super().__init__()
        self.activation = activation  # Store for weight initialization
        
        # Get the appropriate activation function
        act_layer = get_activation(activation)
        
        # First convolution layer
        if small_inputs:
            # For small inputs like CIFAR
            self.features = nn.Sequential(
                nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_init_features),
                act_layer
            )
        else:
            # For larger inputs like ImageNet
            self.features = nn.Sequential(
                nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                act_layer,
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
                bn_size=bn_size,
                use_attention=use_attention,
                reduction_ratio=reduction_ratio,
                dropout_rate=dropout_rate,
                stochastic_depth_prob=stochastic_depth_prob,
                activation=activation
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # Add a transition layer except after the last block
            if i != len(block_config) - 1:
                out_features = int(num_features * compression_rate)
                trans = Transition(
                    num_features, 
                    out_features,
                    use_attention=use_attention,
                    reduction_ratio=reduction_ratio,
                    dropout_rate=dropout_rate,
                    activation=activation
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('act5', get_activation(activation))
        
        # Global pooling
        if use_attention_pooling:
            self.pool = HierarchicalAttentionPooling(num_features, reduction_ratio)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights for better convergence based on activation function
        """
        # Choose appropriate nonlinearity for initialization based on activation
        nonlinearity = 'relu'  # Default
        if self.activation == "swish" or self.activation == "mish":
            # For Swish/Mish, using leaky_relu initialization can help
            nonlinearity = 'leaky_relu'
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
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
        out = self.pool(features)
        
        # Always flatten the output for the classifier
        out = torch.flatten(out, 1)
        
        out = self.classifier(out)
        return out


def DenseNet121(
    num_classes: int = 1000, 
    small_inputs: bool = True, 
    **kwargs
) -> DenseNet:
    """
    Constructs a DenseNet-121 model with attention and improved regularization.
    
    Args:
        num_classes: Number of output classes
        small_inputs: If True, use smaller stem for small inputs like CIFAR
        **kwargs: Additional parameters to pass to the model
    
    Returns:
        DenseNet-121 model
    """
    return DenseNet(
        growth_rate=32, 
        block_config=(6, 12, 24, 16),
        num_init_features=64, 
        num_classes=num_classes,
        small_inputs=small_inputs,
        **kwargs
    )


def DenseNet169(
    num_classes: int = 1000, 
    small_inputs: bool = True, 
    **kwargs
) -> DenseNet:
    """
    Constructs a DenseNet-169 model with attention and improved regularization.
    
    Args:
        num_classes: Number of output classes
        small_inputs: If True, use smaller stem for small inputs like CIFAR
        **kwargs: Additional parameters to pass to the model
    
    Returns:
        DenseNet-169 model
    """
    return DenseNet(
        growth_rate=32, 
        block_config=(6, 12, 32, 32),
        num_init_features=64, 
        num_classes=num_classes,
        small_inputs=small_inputs,
        **kwargs
    )


def DenseNet201(
    num_classes: int = 1000, 
    small_inputs: bool = True, 
    **kwargs
) -> DenseNet:
    """
    Constructs a DenseNet-201 model with attention and improved regularization.
    
    Args:
        num_classes: Number of output classes
        small_inputs: If True, use smaller stem for small inputs like CIFAR
        **kwargs: Additional parameters to pass to the model
    
    Returns:
        DenseNet-201 model
    """
    return DenseNet(
        growth_rate=32, 
        block_config=(6, 12, 48, 32),
        num_init_features=64, 
        num_classes=num_classes,
        small_inputs=small_inputs,
        **kwargs
    ) 