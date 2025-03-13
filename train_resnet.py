#!/usr/bin/env python
"""
Script to train models on a subset of the training data.
This is a wrapper around main.py that simplifies the command-line interface.
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Train a model on a subset of data")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to the configuration file")
    parser.add_argument("--subset", "-s", type=float, required=True,
                        help="Fraction of training data to use (0.0-1.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (optional)")
    parser.add_argument("--batch-size", "-b", type=int, default=None,
                        help="Batch size (optional)")
    parser.add_argument("--epochs", "-e", type=int, default=None,
                        help="Number of epochs (optional)")
    
    args = parser.parse_args()
    
    # Validate subset fraction
    if args.subset <= 0.0 or args.subset > 1.0:
        print("Error: Subset fraction must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, "main.py", "--cfg", args.config, "--subset-fraction", str(args.subset)]
    
    # Add optional arguments
    if args.output:
        cmd.extend(["--output", args.output])
    
    # Add any additional config overrides using the --opts syntax
    extra_opts = []
    if args.batch_size:
        extra_opts.extend(["DATA.BATCH_SIZE", str(args.batch_size)])
    if args.epochs:
        extra_opts.extend(["TRAIN.EPOCHS", str(args.epochs)])
    
    if extra_opts:
        cmd.extend(["--opts"])
        cmd.extend(extra_opts)
    
    # Print the command
    print("Running command:", " ".join(cmd))
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 