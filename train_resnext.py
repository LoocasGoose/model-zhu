#!/usr/bin/env python
"""
Script to train ResNeXt models (ResNeXt29, ResNeXt50, and ResNeXt101) on ImageNet.
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Train ResNeXt models on ImageNet")
    parser.add_argument("--model", "-m", type=str, choices=["resnext29", "resnext50", "resnext101", "all"],
                        default="all", help="Which ResNeXt model to train (default: all)")
    parser.add_argument("--subset", "-s", type=float, default=1.0,
                        help="Fraction of training data to use (0.0-1.0)")
    parser.add_argument("--batch-size", "-b", type=int, default=None,
                        help="Override the batch size from the config")
    parser.add_argument("--epochs", "-e", type=int, default=None,
                        help="Override the number of epochs from the config")
    parser.add_argument("--cardinality", "-c", type=int, default=None,
                        help="Override the cardinality parameter")
    parser.add_argument("--base-width", "-w", type=int, default=None,
                        help="Override the base width parameter")
    parser.add_argument("--data-path", "-d", type=str, default=None,
                        help="Override the path to the ImageNet dataset")
    
    args = parser.parse_args()
    
    # Validate subset fraction
    if args.subset <= 0.0 or args.subset > 1.0:
        print("Error: Subset fraction must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Define which models to train
    if args.model == "all":
        models = ["resnext29", "resnext50", "resnext101"]
    else:
        models = [args.model]
    
    # First check if the base config exists
    base_config = "configs/resnext_base.yaml"
    if not os.path.exists(base_config):
        print(f"Warning: Base configuration file {base_config} not found.")
        print("Model configs may not inherit properly.")
    
    # Train each selected model
    for model in models:
        config_file = f"configs/{model}_imagenet.yaml"
        
        # Check if config file exists
        if not os.path.exists(config_file):
            print(f"Error: Configuration file {config_file} not found")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Training {model.upper()} on ImageNet")
        print(f"{'=' * 80}\n")
        
        # Build command
        cmd = [sys.executable, "main.py", "--cfg", config_file]
        
        # Add subset fraction if specified
        if args.subset < 1.0:
            cmd.extend(["--subset-fraction", str(args.subset)])
        
        # Add any additional config overrides using the --opts syntax
        extra_opts = []
        if args.batch_size:
            extra_opts.extend(["DATA.BATCH_SIZE", str(args.batch_size)])
        if args.epochs:
            extra_opts.extend(["TRAIN.EPOCHS", str(args.epochs)])
        if args.cardinality:
            extra_opts.extend(["MODEL.RESNEXT.CARDINALITY", str(args.cardinality)])
        if args.base_width:
            extra_opts.extend(["MODEL.RESNEXT.BASE_WIDTH", str(args.base_width)])
        if args.data_path:
            extra_opts.extend(["DATA.MEDIUM_IMAGENET_PATH", args.data_path])
        
        if extra_opts:
            cmd.extend(["--opts"])
            cmd.extend(extra_opts)
        
        # Print the command
        print("Running command:", " ".join(cmd))
        
        # Execute the command
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Training {model} failed with exit code {e.returncode}")
            if input("Continue with next model? (y/n): ").lower() != 'y':
                sys.exit(e.returncode)
        except KeyboardInterrupt:
            print(f"Training {model} interrupted by user")
            if input("Continue with next model? (y/n): ").lower() != 'y':
                sys.exit(1)
                
    print("\nTraining complete for all specified ResNeXt models!")

if __name__ == "__main__":
    main() 