from .lenet import LeNet
from .resnet import ResNet18
from models.alexnet import AlexNet
from .resnext import ResNeXt29, ResNeXt50, ResNeXt101


def build_model(config):
    "Model builder."

    model_type = config.MODEL.NAME

    if model_type == 'lenet':
        model = LeNet(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'resnet18':
        model = ResNet18(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'alexnet':
        model = AlexNet(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'resnext29':
        # ResNeXt-29 with 16x64d configuration (optimized for CIFAR)
        cardinality = config.MODEL.RESNEXT.CARDINALITY
        base_width = config.MODEL.RESNEXT.BASE_WIDTH
        # For CIFAR, we typically use different cardinality/width settings
        if cardinality == 32 and base_width == 4:  # If using default values
            cardinality = 16  # Better for CIFAR
            base_width = 64   # Better for CIFAR
            
        # Get optimization parameters
        pruning_rate = getattr(config.MODEL.RESNEXT, 'PRUNING_RATE', 1.0)
        activation = getattr(config.MODEL.RESNEXT, 'ACTIVATION', 'relu')
        use_checkpoint = getattr(config.MODEL.RESNEXT, 'USE_CHECKPOINT', False)
        
        model = ResNeXt29(num_classes=config.MODEL.NUM_CLASSES, 
                          cardinality=cardinality,
                          base_width=base_width,
                          pruning_rate=pruning_rate,
                          activation=activation,
                          use_checkpoint=use_checkpoint)
    elif model_type == 'resnext50':
        # ResNeXt-50 with 32x4d configuration by default
        cardinality = config.MODEL.RESNEXT.CARDINALITY
        base_width = config.MODEL.RESNEXT.BASE_WIDTH
        
        # Get optimization parameters
        pruning_rate = getattr(config.MODEL.RESNEXT, 'PRUNING_RATE', 0.9)  # Slight pruning by default for ResNeXt50
        activation = getattr(config.MODEL.RESNEXT, 'ACTIVATION', 'silu')   # SiLU by default for better performance
        use_checkpoint = getattr(config.MODEL.RESNEXT, 'USE_CHECKPOINT', False)
        
        model = ResNeXt50(num_classes=config.MODEL.NUM_CLASSES,
                          cardinality=cardinality,
                          base_width=base_width,
                          pruning_rate=pruning_rate,
                          activation=activation,
                          use_checkpoint=use_checkpoint)
    elif model_type == 'resnext101':
        # ResNeXt-101 with 32x4d configuration by default
        cardinality = config.MODEL.RESNEXT.CARDINALITY
        base_width = config.MODEL.RESNEXT.BASE_WIDTH
        
        # Get optimization parameters with more aggressive defaults for the largest model
        pruning_rate = getattr(config.MODEL.RESNEXT, 'PRUNING_RATE', 0.8)   # More pruning by default for ResNeXt101
        activation = getattr(config.MODEL.RESNEXT, 'ACTIVATION', 'silu')    # SiLU by default for better performance
        use_checkpoint = getattr(config.MODEL.RESNEXT, 'USE_CHECKPOINT', True)  # Use checkpointing by default
        
        model = ResNeXt101(num_classes=config.MODEL.NUM_CLASSES,
                           cardinality=cardinality,
                           base_width=base_width,
                           pruning_rate=pruning_rate,
                           activation=activation,
                           use_checkpoint=use_checkpoint)
    # elif model_type == 'densenet':
    #     # Use densenet121 by default, but allow command-line override 
    #     # via the --opts feature for different model variations
    #     model_variation = getattr(config, 'VARIATION', '121')
        
    #     # Default parameters that can be overridden via command line options
    #     # Use memory-efficient settings by default
    #     small_inputs = True
    #     use_attention = 'none'  # Disable attention by default to save memory
    #     activation = 'relu'     # Use simpler activation to save memory
    #     attention_pooling = False
    #     stochastic_depth = 0.0
    #     dropout_rate = 0.2      # Add some dropout for regularization
        
    #     # Override defaults if specified via command line options
    #     if hasattr(config, 'DENSENET_ATTENTION'):
    #         use_attention = config.DENSENET_ATTENTION
    #     if hasattr(config, 'DENSENET_ACTIVATION'):
    #         activation = config.DENSENET_ACTIVATION
    #     if hasattr(config, 'DENSENET_ATTENTION_POOLING'):
    #         attention_pooling = config.DENSENET_ATTENTION_POOLING
    #     if hasattr(config, 'DENSENET_STOCHASTIC_DEPTH'):
    #         stochastic_depth = config.DENSENET_STOCHASTIC_DEPTH
    #     if hasattr(config, 'DENSENET_DROPOUT'):
    #         dropout_rate = config.DENSENET_DROPOUT
        
    #     # Select the appropriate DenseNet model based on variation
    #     # For memory issues, we'll override to always use the smaller DenseNet121
    #     # Uncomment the if/elif blocks once memory issues are resolved
        
    #     # Ensure we're using DenseNet121 which has the smallest memory footprint
    #     model = DenseNet121(
    #         num_classes=config.MODEL.NUM_CLASSES,
    #         small_inputs=small_inputs,
    #         use_attention=use_attention,
    #         activation=activation,
    #         use_attention_pooling=attention_pooling,
    #         stochastic_depth_prob=stochastic_depth,
    #         dropout_rate=dropout_rate   # Add dropout parameter
    #     )
        
    #     # Log the model configuration to help with debugging
    #     print(f"Using DenseNet121 with: activation={activation}, attention={use_attention}, "
    #           f"dropout={dropout_rate}, img_size={config.DATA.IMG_SIZE}, batch_size={config.DATA.BATCH_SIZE}")
        
    #     # Comment out these larger models to conserve memory
    #     # If memory issues are resolved, you can switch to using these models
    #     """
    #     if model_variation == '169':
    #         model = DenseNet169(
    #             num_classes=config.MODEL.NUM_CLASSES,
    #             small_inputs=small_inputs,
    #             use_attention=use_attention,
    #             activation=activation,
    #             use_attention_pooling=attention_pooling,
    #             stochastic_depth_prob=stochastic_depth,
    #             dropout_rate=dropout_rate
    #         )
    #     elif model_variation == '201':
    #         model = DenseNet201(
    #             num_classes=config.MODEL.NUM_CLASSES,
    #             small_inputs=small_inputs,
    #             use_attention=use_attention,
    #             activation=activation,
    #             use_attention_pooling=attention_pooling,
    #             stochastic_depth_prob=stochastic_depth,
    #             dropout_rate=dropout_rate
    #         )
    #     else:  # Default to 121
    #         model = DenseNet121(
    #             num_classes=config.MODEL.NUM_CLASSES,
    #             small_inputs=small_inputs,
    #             use_attention=use_attention,
    #             activation=activation,
    #             use_attention_pooling=attention_pooling,
    #             stochastic_depth_prob=stochastic_depth,
    #             dropout_rate=dropout_rate
    #         )
    #     """
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
 