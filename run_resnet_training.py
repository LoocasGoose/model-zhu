#!/usr/bin/env python3
"""
Script to run the unified ResNet training with various configurations.
This provides easy commands to train different models with optimized settings.
"""
import os
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, description=None):
    """Run a command with detailed output"""
    if description:
        print(f"\n\n{'=' * 80}")
        print(f"Running: {description}")
        print(f"{'=' * 80}\n")
    
    print(f"Command: {cmd}\n")
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    end_time = time.time()
    
    print(f"\nCommand completed in {end_time - start_time:.2f} seconds with return code {result.returncode}\n")
    return result.returncode


def main():
    parser = argparse.ArgumentParser("ResNet Training Script Runner")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["resnet18", "resnetv2_18", "resnetv2_50", 
                                "all", "eval", "compare", "memory_efficient"],
                        help="Training mode to run")
    parser.add_argument("--output", type=str, default="output",
                        help="Base output directory")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--config", type=str, default=None,
                        help="Override default config for the selected mode")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Base command template
    base_cmd = f"python train_unified_resnet.py --gpu {args.gpu}"
    
    # Add batch size and epochs if specified
    if args.batch_size:
        base_cmd += f" --batch-size {args.batch_size}"
    if args.epochs:
        base_cmd += f" --epochs {args.epochs}"
    
    # Run the selected mode
    if args.mode == "resnet18":
        # Original ResNet18 from resnet.py
        config = args.config or "configs/resnet18_medium_imagenet.yaml"
        cmd = f"{base_cmd} --model-type resnet --model-variant resnet18 --cfg {config} --output {args.output}/original_resnet18"
        run_command(cmd, "Training original ResNet18 from resnet.py")
        
    elif args.mode == "resnetv2_18":
        # ResNetV2-18 from resnetv2.py
        config = args.config or "configs/resnet18_imagenet.yaml"
        cmd = f"{base_cmd} --model-type resnetv2 --model-variant resnet18 --cfg {config} --output {args.output}/resnetv2_18"
        run_command(cmd, "Training ResNetV2-18 from resnetv2.py")
        
    elif args.mode == "resnetv2_50":
        # ResNetV2-50 from resnetv2.py
        config = args.config or "configs/resnet50_imagenet.yaml"
        cmd = f"{base_cmd} --model-type resnetv2 --model-variant resnet50 --cfg {config} --output {args.output}/resnetv2_50"
        run_command(cmd, "Training ResNetV2-50 from resnetv2.py")
        
    elif args.mode == "all":
        # Run all model variants
        configs = {
            "resnet18": args.config or "configs/resnet18_medium_imagenet.yaml",
            "resnetv2_18": args.config or "configs/resnet18_imagenet.yaml",
            "resnetv2_50": args.config or "configs/resnet50_imagenet.yaml"
        }
        
        # Original ResNet18
        cmd = f"{base_cmd} --model-type resnet --model-variant resnet18 --cfg {configs['resnet18']} --output {args.output}/original_resnet18"
        run_command(cmd, "Training original ResNet18 from resnet.py")
        
        # ResNetV2-18
        cmd = f"{base_cmd} --model-type resnetv2 --model-variant resnet18 --cfg {configs['resnetv2_18']} --output {args.output}/resnetv2_18"
        run_command(cmd, "Training ResNetV2-18 from resnetv2.py")
        
        # ResNetV2-50
        cmd = f"{base_cmd} --model-type resnetv2 --model-variant resnet50 --cfg {configs['resnetv2_50']} --output {args.output}/resnetv2_50"
        run_command(cmd, "Training ResNetV2-50 from resnetv2.py")
        
    elif args.mode == "eval":
        # Evaluate trained models
        
        # Paths to the best model checkpoints
        checkpoints = {
            "resnet18": f"{args.output}/original_resnet18/model_best.pth",
            "resnetv2_18": f"{args.output}/resnetv2_18/model_best.pth",
            "resnetv2_50": f"{args.output}/resnetv2_50/model_best.pth"
        }
        
        # Check which models exist and evaluate them
        for model_type, model_variant, checkpoint_path in [
            ("resnet", "resnet18", checkpoints["resnet18"]),
            ("resnetv2", "resnet18", checkpoints["resnetv2_18"]),
            ("resnetv2", "resnet50", checkpoints["resnetv2_50"])
        ]:
            if os.path.exists(checkpoint_path):
                config = "configs/resnet18_medium_imagenet.yaml" if model_type == "resnet" else \
                         "configs/resnet18_imagenet.yaml" if model_variant == "resnet18" else \
                         "configs/resnet50_imagenet.yaml"
                
                cmd = f"{base_cmd} --model-type {model_type} --model-variant {model_variant} --cfg {config} --eval --resume {checkpoint_path}"
                run_command(cmd, f"Evaluating {model_type}_{model_variant}")
            else:
                print(f"Checkpoint not found for {model_type}_{model_variant} at {checkpoint_path}")
    
    elif args.mode == "compare":
        # Run a comparison with fixed settings across all models
        # Use the same batch size, epochs, and other settings for fair comparison
        comparison_settings = "--batch-size 32 --epochs 10 --validate-freq 2"
        
        # Original ResNet18
        cmd = f"{base_cmd} {comparison_settings} --model-type resnet --model-variant resnet18 --cfg configs/resnet18_medium_imagenet.yaml --output {args.output}/compare_original_resnet18"
        run_command(cmd, "Comparing original ResNet18")
        
        # ResNetV2-18
        cmd = f"{base_cmd} {comparison_settings} --model-type resnetv2 --model-variant resnet18 --cfg configs/resnet18_imagenet.yaml --output {args.output}/compare_resnetv2_18"
        run_command(cmd, "Comparing ResNetV2-18")
        
        # ResNetV2-50
        cmd = f"{base_cmd} {comparison_settings} --model-type resnetv2 --model-variant resnet50 --cfg configs/resnet50_imagenet.yaml --output {args.output}/compare_resnetv2_50"
        run_command(cmd, "Comparing ResNetV2-50")
    
    elif args.mode == "memory_efficient":
        # Run models with memory efficiency optimizations
        memory_settings = "--grad-accum-steps 2"  # Use gradient accumulation
        
        # Original ResNet18 with memory optimizations
        cmd = f"{base_cmd} {memory_settings} --model-type resnet --model-variant resnet18 --cfg configs/resnet18_medium_imagenet.yaml --output {args.output}/memory_efficient_original_resnet18"
        run_command(cmd, "Training memory-efficient original ResNet18")
        
        # ResNetV2-50 with memory optimizations
        cmd = f"{base_cmd} {memory_settings} --model-type resnetv2 --model-variant resnet50 --cfg configs/resnet50_imagenet.yaml --output {args.output}/memory_efficient_resnetv2_50"
        run_command(cmd, "Training memory-efficient ResNetV2-50")


if __name__ == "__main__":
    main() 