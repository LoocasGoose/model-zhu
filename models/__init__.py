from .resnet import ResNet18
import torch.nn as nn

# Function to build model from config
def build_model(config):
    """
    Build model based on configuration
    
    Args:
        config: Configuration object with model specifications
        
    Returns:
        model: Instantiated model
    """
    model_type = config.MODEL.NAME
    
    if model_type == 'resnet18':
        # Check if we should enable gradient checkpointing
        enable_checkpoint = getattr(config, 'ENABLE_CHECKPOINT', False)
        # Create ResNet18 model
        model = ResNet18(
            num_classes=config.MODEL.NUM_CLASSES,
            enable_checkpoint=enable_checkpoint
        )
    elif model_type == 'alexnet':
        # Simple AlexNet implementation
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, config.MODEL.NUM_CLASSES),
        )
    elif model_type == 'lenet':
        # Simple LeNet implementation
        model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * ((config.DATA.IMG_SIZE - 8) // 4) ** 2, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, config.MODEL.NUM_CLASSES),
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    return model
