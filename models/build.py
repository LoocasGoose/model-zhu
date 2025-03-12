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
    elif model_type == 'densenet':
        # Get the densenet type
        densenet_type = config.MODEL.get('TYPE', '121')
        attention = config.MODEL.get('ATTENTION', 'se')
        activation = config.MODEL.get('ACTIVATION', 'swish')
        attention_pooling = config.MODEL.get('ATTENTION_POOLING', False)
        stochastic_depth = config.MODEL.get('STOCHASTIC_DEPTH', 0.0)
        
        # Select the appropriate DenseNet model based on type
        if densenet_type == '121':
            model = DenseNet121(
                num_classes=config.MODEL.NUM_CLASSES,
                small_inputs=True,
                use_attention=attention,
                activation=activation,
                use_attention_pooling=attention_pooling,
                stochastic_depth_prob=stochastic_depth
            )
        elif densenet_type == '169':
            model = DenseNet169(
                num_classes=config.MODEL.NUM_CLASSES,
                small_inputs=True,
                use_attention=attention,
                activation=activation,
                use_attention_pooling=attention_pooling,
                stochastic_depth_prob=stochastic_depth
            )
        elif densenet_type == '201':
            model = DenseNet201(
                num_classes=config.MODEL.NUM_CLASSES,
                small_inputs=True,
                use_attention=attention,
                activation=activation,
                use_attention_pooling=attention_pooling,
                stochastic_depth_prob=stochastic_depth
            )
        else:
            raise ValueError(f"Invalid densenet type: {densenet_type}")
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
 