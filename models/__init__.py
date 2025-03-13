from .resnext import ResNeXt, ResNeXtBlock, ResNeXt29, ResNeXt50, ResNeXt101

# Function to build model from config
def build_model(config):
    """
    Build the specific model based on configuration.
    
    Args:
        config: Configuration object that contains model specifications
        
    Returns:
        Instantiated model
    """
    model_type = config.MODEL.NAME
    
    # Set default values for parameters that might not exist in all config files
    drop_rate = 0.0
    if hasattr(config.MODEL.RESNEXT, 'DROP_RATE'):
        drop_rate = config.MODEL.RESNEXT.DROP_RATE
    elif hasattr(config.MODEL, 'DROP_RATE'):  # Fallback to the old location
        drop_rate = config.MODEL.DROP_RATE
    
    if model_type == "resnext29":
        model = ResNeXt29(
            num_classes=config.DATA.NUM_CLASSES if hasattr(config.DATA, 'NUM_CLASSES') else 200,
            cardinality=config.MODEL.RESNEXT.CARDINALITY,
            base_width=config.MODEL.RESNEXT.BASE_WIDTH,
            pruning_rate=config.MODEL.RESNEXT.PRUNING_RATE,
            activation=config.MODEL.RESNEXT.ACTIVATION,
            use_checkpoint=config.MODEL.RESNEXT.USE_CHECKPOINT,
            drop_rate=drop_rate,
            small_input=getattr(config.DATA, 'SMALL_INPUT', True)
        )
    elif model_type == "resnext50":
        model = ResNeXt50(
            num_classes=config.DATA.NUM_CLASSES if hasattr(config.DATA, 'NUM_CLASSES') else 200,
            cardinality=config.MODEL.RESNEXT.CARDINALITY,
            base_width=config.MODEL.RESNEXT.BASE_WIDTH,
            pruning_rate=config.MODEL.RESNEXT.PRUNING_RATE,
            activation=config.MODEL.RESNEXT.ACTIVATION,
            use_checkpoint=config.MODEL.RESNEXT.USE_CHECKPOINT,
            drop_rate=drop_rate,
            small_input=getattr(config.DATA, 'SMALL_INPUT', False)
        )
    elif model_type == "resnext101":
        model = ResNeXt101(
            num_classes=config.DATA.NUM_CLASSES if hasattr(config.DATA, 'NUM_CLASSES') else 200,
            cardinality=config.MODEL.RESNEXT.CARDINALITY,
            base_width=config.MODEL.RESNEXT.BASE_WIDTH,
            pruning_rate=config.MODEL.RESNEXT.PRUNING_RATE,
            activation=config.MODEL.RESNEXT.ACTIVATION,
            use_checkpoint=config.MODEL.RESNEXT.USE_CHECKPOINT,
            drop_rate=drop_rate,
            small_input=getattr(config.DATA, 'SMALL_INPUT', False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
