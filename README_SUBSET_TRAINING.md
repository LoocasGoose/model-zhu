# Training on a Subset of Data

This feature allows you to train models on a subset of the training data, which can be useful for:

1. Quick experimentation and debugging
2. Testing model architecture changes without waiting for a full training run
3. Studying model behavior with limited data

## How to Use

You can specify the fraction of training data to use in three ways:

### 1. Command Line Argument

```bash
python main.py --cfg configs/resnet18_base.yaml --subset-fraction 0.1
```

This will train the model using only 10% of the training data.

### 2. Configuration File

You can also modify the configuration files directly by adding or changing the `SUBSET_FRACTION` parameter:

```yaml
DATA:
  BATCH_SIZE: 1024
  DATASET: "cifar10"
  IMG_SIZE: 32
  SUBSET_FRACTION: 0.2  # Use 20% of the training data
```

### 3. Using the Provided Scripts

For convenience, two example scripts are provided:

- `train_resnet18_subset.sh`: Trains ResNet18 on CIFAR-10 using 10% of the data
- `train_resnet18_medium_imagenet_subset.sh`: Trains ResNet18 on Medium ImageNet using 20% of the data

On Windows, run these scripts as follows:
```
.\train_resnet18_subset.sh
```

On Linux/Mac:
```bash
./train_resnet18_subset.sh
```

## Implementation Details

The subset selection is performed randomly at the beginning of each training run. The same subset is used throughout the entire training process for that run.

The code will print the actual number of samples used and the percentage of the original dataset at the beginning of training.

## Notes

- The validation and test sets are always used in full (not subsampled).
- Setting `SUBSET_FRACTION` to 1.0 (the default) will use the entire training dataset.
- Setting `SUBSET_FRACTION` too low (e.g., 0.01 for just 1% of the data) might result in poor model performance due to insufficient training examples. 