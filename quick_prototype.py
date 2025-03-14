#!/usr/bin/env python3
"""
Quick prototyping script for ResNetV2 models.
This script provides pre-configured settings for rapid prototyping.
"""
import argparse
import subprocess
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Quick prototype ResNetV2 training')
    parser.add_argument('--config', '-c', type=str, default='configs/resnet18v2.yaml',
                        help='Path to config file (default: configs/resnet18v2.yaml)')
    parser.add_argument('--size', '-s', type=str, default='xs',
                        choices=['xs', 's', 'm', 'l', 'xl'],
                        help='Subset size: xs=1%%, s=5%%, m=10%%, l=25%%, xl=50%%')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', '-b', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='Custom name for output directory')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (default: 0)')
    parser.add_argument('--workers', '-w', type=int, default=32,
                        help='Number of data loading workers (default: 32)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define subset sizes
    subset_sizes = {
        'xs': 0.01,  # 1% of dataset
        's': 0.05,   # 5% of dataset
        'm': 0.10,   # 10% of dataset
        'l': 0.25,   # 25% of dataset
        'xl': 0.50,  # 50% of dataset
    }
    
    subset = subset_sizes[args.size]
    
    # Construct output directory if custom name provided
    output_args = []
    if args.name:
        output_dir = f"output/{args.name}_proto_{args.size}_{args.epochs}ep"
        output_args = ["--output", output_dir]
    
    # Build command
    cmd = [
        "python", "train_resnetv2.py",
        "--cfg", args.config,
        "--quick-train",
        "--quick-epochs", str(args.epochs),
        "--subset", str(subset),
        "--batch-size", str(args.batch_size),
        "--gpu", str(args.gpu),
        "--workers", str(args.workers),
    ] + output_args
    
    # Print configuration
    print("=" * 80)
    print(f"Starting quick prototype training with:")
    print(f"  Config:     {args.config}")
    print(f"  Subset:     {args.size} ({subset*100:.1f}% of data)")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  GPU:        {args.gpu}")
    print(f"  Workers:    {args.workers}")
    if args.name:
        print(f"  Output dir: {output_dir}")
    print("=" * 80)
    
    # Execute command
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        return e.returncode
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 