import argparse
import sys
from config import get_config

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
    
if __name__ == "__main__":
    main() 