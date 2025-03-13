import argparse
import sys
import torch
from config import get_config
from models import build_model

def main():
    args = argparse.Namespace(
        cfg='configs/resnet18_medium_imagenet.yaml', 
        opts=None, 
        batch_size=None, 
        data_path=None, 
        resume=None, 
        output='output', 
        subset_fraction=1.0, 
        eval=False
    )
    
    config = get_config(args)
    
    print("Config loaded successfully!")
    print(f"Model name: {config.MODEL.NAME}")
    print(f"Number of classes: {config.MODEL.NUM_CLASSES}")
    print(f"Batch size: {config.DATA.BATCH_SIZE}")
    print(f"Image size: {config.DATA.IMG_SIZE}")
    
    # Try building the model
    try:
        model = build_model(config)
        print(f"Model built successfully: {type(model).__name__}")
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Try moving the model to device
        model = model.to(device)
        print(f"Model moved to {device} successfully")
        
        # Count parameters
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {n_parameters / 1e6:.2f}M")
    except Exception as e:
        print(f"Error building model: {e}")
    
if __name__ == "__main__":
    main() 