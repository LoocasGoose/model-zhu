from .lenet import LeNet
from .resnet import ResNet18
from models.alexnet import AlexNet
from .densenet import DenseNet121, DenseNet169, DenseNet201


def build_model(config):
    "Model builder."

    model_type = config.MODEL.NAME

    if model_type == 'lenet':
        model = LeNet(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'resnet18':
        model = ResNet18(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'alexnet':
        model = AlexNet(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'densenet121':
        model = DenseNet121(
            num_classes=config.MODEL.NUM_CLASSES, 
            small_inputs=config.MODEL.get('SMALL_INPUTS', True),
            use_attention=config.MODEL.get('ATTENTION', 'se'),
            reduction_ratio=config.MODEL.get('REDUCTION_RATIO', 16),
            dropout_rate=config.MODEL.get('DROP_RATE', 0.2),
            stochastic_depth_prob=config.MODEL.get('STOCHASTIC_DEPTH_PROB', 0.0),
            activation=config.MODEL.get('ACTIVATION', 'swish')
        )
    elif model_type == 'densenet169':
        model = DenseNet169(
            num_classes=config.MODEL.NUM_CLASSES, 
            small_inputs=config.MODEL.get('SMALL_INPUTS', True),
            use_attention=config.MODEL.get('ATTENTION', 'se'),
            reduction_ratio=config.MODEL.get('REDUCTION_RATIO', 16),
            dropout_rate=config.MODEL.get('DROP_RATE', 0.2),
            stochastic_depth_prob=config.MODEL.get('STOCHASTIC_DEPTH_PROB', 0.0),
            activation=config.MODEL.get('ACTIVATION', 'swish')
        )
    elif model_type == 'densenet201':
        model = DenseNet201(
            num_classes=config.MODEL.NUM_CLASSES, 
            small_inputs=config.MODEL.get('SMALL_INPUTS', True),
            use_attention=config.MODEL.get('ATTENTION', 'se'),
            reduction_ratio=config.MODEL.get('REDUCTION_RATIO', 16),
            dropout_rate=config.MODEL.get('DROP_RATE', 0.2),
            stochastic_depth_prob=config.MODEL.get('STOCHASTIC_DEPTH_PROB', 0.0),
            activation=config.MODEL.get('ACTIVATION', 'swish')
        )
    # elif model_type == 'resnet34':
    #     model = ResNet34(num_classes=config.MODEL.NUM_CLASSES)
    # elif model_type == 'resnet50':
    #     model = ResNet50(num_classes=config.MODEL.NUM_CLASSES)
    # elif model_type == 'resnet101':
    #     model = ResNet101(num_classes=config.MODEL.NUM_CLASSES)
    # elif model_type == 'resnet152':
    #     model = ResNet152(num_classes=config.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    
    return model
 