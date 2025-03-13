#!/bin/bash
# Optimized training script for ResNeXt models on RTX 2080

# Parse arguments
COMPILE=0
TRACK=0
PROFILE=0
BATCH_SIZE=64

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --compile)
      COMPILE=1
      shift
      ;;
    --track)
      TRACK=1
      shift
      ;;
    --profile)
      PROFILE=1
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      # Store all other arguments to pass to main.py
      MAIN_ARGS="$MAIN_ARGS $1"
      shift
      ;;
  esac
done

# Set environment variables for performance
export CUDA_VISIBLE_DEVICES=0  # Use only the first GPU if there are multiple
export OMP_NUM_THREADS=4  # Set OpenMP threads to avoid CPU oversubscription
export MKL_NUM_THREADS=4  # Limit MKL threads to match OMP threads

# Additional PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"  # Reduce memory fragmentation
export TORCH_CUDNN_V8_API_ENABLED=1  # Enable cuDNN v8 API if available
export TORCH_USE_RTLD_GLOBAL=1  # Improve dynamic library loading
export MAGMA_NUM_THREADS=2  # Optimize MAGMA library thread count

# Clear CUDA cache before running
python -c "import torch; torch.cuda.empty_cache()"

# Function to enable performance tracking
setup_tracking() {
  mkdir -p ./logs/perf_stats
  # Create an empty file to track training times
  echo "epoch,time_seconds,throughput,loss,accuracy,memory_mb" > ./logs/perf_stats/training_metrics.csv
}

# Start time tracking for entire training run
start_time=$(date +%s)

# Pre-compile the model with TorchScript for faster execution (if requested)
if [ $COMPILE -eq 1 ]; then
  echo "Precompiling model with TorchScript (experimental)..."
  python -c "
import torch
import sys
sys.path.append('.')
from models.resnext import ResNeXt29
model = ResNeXt29(num_classes=200, small_input=True)
model = model.cuda().to(memory_format=torch.channels_last)
example = torch.rand(1, 3, 224, 224).cuda().to(memory_format=torch.channels_last)
traced_model = torch.jit.trace(model, example)
traced_model.save('resnext29_compiled.pt')
"
fi

# Create performance tracking logs if enabled
if [ $TRACK -eq 1 ]; then
  setup_tracking
fi

# Determine optimal batch size based on mixed precision
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
    print(f'Using batch size: $BATCH_SIZE')
"

# Run training with optimized settings
echo "Starting optimized ResNeXt training..."
echo "Running with batch size: $BATCH_SIZE"
python main.py \
  --cfg configs/resnext29_imagenet.yaml \
  --batch-size $BATCH_SIZE \
  $MAIN_ARGS # Pass any additional arguments to the script

# Print training time at the end
end_time=$(date +%s)
training_time=$((end_time - start_time))
echo "Total training time: $training_time seconds ($(($training_time / 60)) minutes)"

# Optional: Profile the training if requested
if [ $PROFILE -eq 1 ]; then
  echo "Profiling training for optimization insights..."
  python -m torch.utils.bottleneck main.py --cfg configs/resnext29_imagenet.yaml --batch-size 32
fi 