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
        # Use densenet121 by default, but allow command-line override 
        # via the --opts feature for different model variations
        model_variation = getattr(config, 'VARIATION', '121')
        
        # Default parameters that can be overridden via command line options
        small_inputs = True
        use_attention = 'se'
        activation = 'swish'
        attention_pooling = False
        stochastic_depth = 0.0
        
        # Override defaults if specified via command line options
        if hasattr(config, 'DENSENET_ATTENTION'):
            use_attention = config.DENSENET_ATTENTION
        if hasattr(config, 'DENSENET_ACTIVATION'):
            activation = config.DENSENET_ACTIVATION
        if hasattr(config, 'DENSENET_ATTENTION_POOLING'):
            attention_pooling = config.DENSENET_ATTENTION_POOLING
        if hasattr(config, 'DENSENET_STOCHASTIC_DEPTH'):
            stochastic_depth = config.DENSENET_STOCHASTIC_DEPTH
        
        # Select the appropriate DenseNet model based on variation
        if model_variation == '169':
            model = DenseNet169(
                num_classes=config.MODEL.NUM_CLASSES,
                small_inputs=small_inputs,
                use_attention=use_attention,
                activation=activation,
                use_attention_pooling=attention_pooling,
                stochastic_depth_prob=stochastic_depth
            )
        elif model_variation == '201':
            model = DenseNet201(
                num_classes=config.MODEL.NUM_CLASSES,
                small_inputs=small_inputs,
                use_attention=use_attention,
                activation=activation,
                use_attention_pooling=attention_pooling,
                stochastic_depth_prob=stochastic_depth
            )
        else:  # Default to 121
            model = DenseNet121(
                num_classes=config.MODEL.NUM_CLASSES,
                small_inputs=small_inputs,
                use_attention=use_attention,
                activation=activation,
                use_attention_pooling=attention_pooling,
                stochastic_depth_prob=stochastic_depth
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
 