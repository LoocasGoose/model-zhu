# NMEP Homework 3: Computer Vision Model Zhu

In this homework, you will be implementing a few popular computer vision models, and training them on both CIFAR-10 and on a custom dataset we created. You will be using PyTorch for this homework.

You will be using a medium-sized repository which mimics that of a standard codebase which you might find for modern projects. 
Don't be intimidated!
We will walk you through all of the parts of it, and hopefully after this homework you will be more confident working with codebases like this. We believe this is a realistic representation of what you may do in the future, and we hope you will find it useful.

We would recommend you first set up the repository ASAP on honeydew and try running it out of the box to see how it trains, and only afterwards focus on understanding all parts of the code. 
For your benefit, this codebase works out of the box, and you should be able to train a model on CIFAR-10 with no changes. Throughout the assignment, you will need to make some changes to `models/alexnet.py` and `models/resnet.py`, for which you will find the provided implementations of other models in `models/` to be helpful.

All of the assignment details are provided in [`WORKSHEET.md`](WORKSHEET.md) - you will need to fill in some answers and make code changes. We recommend getting through this [`README.md`](README.md) file first and then doing the worksheet.

Best of luck, and we hope you enjoy it!

## Setup 

To get started, you will need to clone the repository and install the dependencies, preferably in a conda environment. Standard instructions are provided below.

```bash
git clone git@github.com:mlberkeley/fa24-nmep-hw2.git
cd fa24-nmep-hw2
conda env create -f env.yml
conda activate vision-zoo
CUDA_VISIBLE_DEVICES=1 python main.py --cfg=configs/lenet_base.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --cfg=configs/alexnet.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/densenet.yaml --data_path /honey/nmep/medium-imagenet-96.hdf5 --batch_size 32 --model_type 169 --attention cbam --activation mish --epochs 50

import torch
print(f'CUDA available: {torch.cuda.is_available()}')
```

This should begin a download and training of a LeNet(ish) model on CIFAR-10. You should see all of the output files in ```output/lenet```, but you can specify exactly where in the configs (more on that in a second).

## Overview of Project Structure

The project is organized roughly as follows:

```bash
configs/            # contains all of the configs for the project
  resnet/           # you can organize configs by model
  ...
data/               # contains all of the data related code
  build.py          # contains the data loader builder
  datasets.py       # where dataset loaders are defined
  ...
models/             # contains all of the model related code
  build.py          # contains the model builder
  resnet.py         # ResNet definition
  ...
utils/              # misc. utils for the project
  ... 
config.py           # contains the config parser; define defaults here!!
main.py             # main training loop
optimizer.py        # optimizer definition
```

You'll notice that the main subfolders all have a ```build.py``` file. This is a common pattern in codebases, and is used to build the model and data loaders using the configs. Generally all the config parameters are handled in the build files, which then call the appropriate class to build the model or data loader. They're kind of a liaison between the configs and the actual code, so that the code can be written free of any config dependencies.

## Configs

Speaking of configs, most projects you'll come across will use a config system to specify hyperparameters and other settings. This is a very common practice, and is used in many of the projects you'll see in the future. We've provided a simple config parser for you to use, which you can find in ```config.py```. You can see how it's used in ```main.py```, where we parse the config and then pass it to the model and data loader builders. Notably, configs are defined and given defaults in ```config.py```, and then can be overridden using yaml files in ```configs/```. This particular system is nested, so for example your configs will look something like this. 

```yaml
# configs/resnet18_cifar.yaml
...
MODEL:
  NAME: resnet18
  NUM_CLASSES: 10
  ...
DATA:
  BATCH_SIZE: 128
  DATASET: cifar10
    ...
```

You'll need to chase them down in the code to understand the exact impact of the settings, but these are useful because they allow you to easily change hyperparameters and settings without having to change the code.
Plus, for experimentation, it's nice to be able to keep track of all of the settings you used for a particular run and have everything you need to reproduce them whenever you want.

However for when you're hacking or just testing things quickly, it's useful to not have to create a new config for everything. Hence we've also provided the option of using a few command line arguments to override the configs. You can see how this is done in ```main.py```, where we parse the command line arguments and then override the configs with them. Throw these together in a shell script to keep track of everything, and you're good to go!

## Tips

Don't try to understand everything at once, it's daunting! Treat this like you would a large class project or a software engineering project, and work in small chunks (it's why we've cleanly factored the code into modules). Ask questions, don't be afraid to test things out in jupyter notebooks or use the pdb debugger (```breakpoint()``` or ```import pdb; pdb.set_trace()```). These are all good skills to learn to become a great machine learning engineer.

# Notes
python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 8 DATA.BATCH_SIZE 4096

python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 8 DATA.BATCH_SIZE 16384

python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 8 DATA.BATCH_SIZE 16384 TRAIN.EPOCHS 10

python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 16 MODEL.NUM_CLASSES 10 DATA.BATCH_SIZE 16384 TRAIN.EPOCHS 20

python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 16 MODEL.NUM_CLASSES 10 DATA.BATCH_SIZE 4096 TRAIN.EPOCHS 20 TRAIN.LR 0.001 TRAIN.OPTIMIZER.MOMENTUM 0.95

python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 16 MODEL.NUM_CLASSES 10 DATA.BATCH_SIZE 16384 TRAIN.EPOCHS 20 TRAIN.LR 0.01 TRAIN.OPTIMIZER.MOMENTUM 0.95

python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 16 MODEL.NUM_CLASSES 10 DATA.BATCH_SIZE 65536 TRAIN.EPOCHS 20 TRAIN.LR 0.01 TRAIN.OPTIMIZER.MOMENTUM 0.95

python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 20 DATA.BATCH_SIZE 64 MODEL.NUM_CLASSES 10 TRAIN.EPOCHS 20 TRAIN.LR 0.01 TRAIN.OPTIMIZER.MOMENTUM 0.95

python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 20 DATA.BATCH_SIZE 81920 MODEL.NUM_CLASSES 10 TRAIN.EPOCHS 20 TRAIN.LR 0.09 TRAIN.OPTIMIZER.MOMENTUM 0.9

python main.py --cfg=configs/lenet_base.yaml --opts DATA.NUM_WORKERS 12 DATA.BATCH_SIZE 49152 MODEL.NUM_CLASSES 10 TRAIN.EPOCHS 20 TRAIN.LR 1.0 TRAIN.OPTIMIZER.MOMENTUM 0.9

16384*3 = 49152

python main.py --cfg=configs/lenet_base.yaml --opts DATA.BATCH_SIZE 32 TRAIN.LR 0.002640329447930566 TRAIN.OPTIMIZER.MOMENTUM 0.9858934559897461 MODEL.NUM_CLASSES 10 TRAIN.EPOCHS 20 DATA.NUM_WORKERS 32

Trial 62 finished with value: 66.68 and parameters: {'learning_rate': 0.007735950805309332, 'momentum': 0.8569584037146034, 'batch_size': 32, 'activation': 'sigmoid', 'weight_decay': 2.5374310287615207e-10}. Best is trial 62 with value: 66.68.



set PYTORCH_NO_CUDA_MEMORY_CACHING=1
set LRU_CACHE_CAPACITY=1

## initalize jupyter notebook
lucas.gu@honeydew:~$ cd ~/nmep
lucas.gu@honeydew:~/nmep$ cd ~/nmep/model-zhu
lucas.gu@honeydew:~/nmep/model-zhu$ cd model-zhu
lucas.gu@honeydew:~/nmep/model-zhu/model-zhu$ touch data/HDF5_visualizer.ipynb
lucas.gu@honeydew:~/nmep/model-zhu/model-zhu$ source ~/miniconda3/bin/activate
(base) lucas.gu@honeydew:~/nmep/model-zhu/model-zhu$ jupyter notebook --no-browser --port=8888

on sep terminal: (base) PS C:\Users\zhiwe> ssh -L 8888:localhost:8888 lucas.gu@honeydew

Alexnet: 
Epoch 20, Max accuracy: 82.44%: python main.py --cfg=configs/alexnet.yaml
Epoch 20, Max accuracy: : L.DROP_RATE 0.4140757605926737
CUDA_VISIBLE_DEVICES=5 python main.py --cfg=configs/alexnet.yaml --opts TRAIN.LR 0.004584093674528438 DATA.BATCH_SIZE 128 MODE


Went down a rabbit hole tuning hyperparameters for the past 5 hours or so - wrote a hyperparameter tuning script and tested ~60 different combinations (can be increased). From my testing, this is one of the best intialization for lenet_base.yaml:
```
python main.py --cfg=configs/lenet_base.yaml --opts DATA.BATCH_SIZE 64 TRAIN.LR 0.007568304298229739 TRAIN.OPTIMIZER.MOMENTUM 0.9540690887792035 MODEL.NUM_CLASSES 10 TRAIN.EPOCHS 20 DATA.NUM_WORKERS 6
```
Got 68% accuracy after 20 epochs (approaching the limit of 70% validation accuracy for a very basic lenet).

I've attached the hyperparameter tuning script if you want to find the best hyperparameters for yourself. I used optuna hyperparameter optimization framework and Hyperband pruning to speed up the process. More details are in the script.

(I crashed my computer 6+ times cuz memory allocation issues yippee)





# DenseNet with Attention Mechanisms

This repository contains an implementation of DenseNet with various attention mechanisms and training utilities.

## Model Features

The DenseNet implementation includes:

- Multiple model configurations (DenseNet-121, DenseNet-169, DenseNet-201)
- Attention mechanisms (SE, CBAM)
- Advanced activation functions (Swish, Mish)
- Stochastic depth for improved regularization
- Attention pooling options

## Dataset

The model is designed to work with the medium-imagenet dataset located at:
```
/honey/nmep/medium-imagenet-96.hdf5
```

## Quick Start

To train a DenseNet model using the configuration system:

```bash
python main.py --cfg configs/densenet.yaml
```

This command loads the configuration from the YAML file, sets up the model, and starts training.

## Configuration Files

The configuration files are stored in the `configs/` directory and use YAML format. You can create different configuration files for different experiments.

### Example Configuration

```yaml
# Dataset Configuration
data_path: '/honey/nmep/medium-imagenet-96.hdf5'
val_split: 0.1
num_workers: 4

# Model Configuration
model_type: 'densenet'
model_size: '121'       # Options: 121, 169, 201
attention: 'cbam'       # Options: se, cbam, none
activation: 'mish'      # Options: swish, mish, relu
attention_pooling: true
stochastic_depth: 0.1

# Training Configuration
batch_size: 64
epochs: 100
learning_rate: 0.001
min_lr: 0.000001
weight_decay: 0.0001
scheduler: 'cosine'     # Options: cosine, plateau, none
early_stopping_patience: 10

# Other Parameters
seed: 42
checkpoint_dir: 'checkpoints'
log_dir: 'logs'
checkpoint_freq: 5
```

## Customizing Configurations

You can create your own configuration files by copying and modifying the example above. Here are the key parameters you can adjust:

### Dataset Parameters

- `data_path`: Path to the HDF5 dataset file
- `val_split`: Proportion of data to use for validation (0.0 to 1.0)
- `num_workers`: Number of data loading workers

### Model Parameters

- `model_type`: Always set to 'densenet' for DenseNet models
- `model_size`: DenseNet architecture size (121, 169, or 201)
- `attention`: Attention mechanism (se, cbam, none)
- `activation`: Activation function (swish, mish, relu)
- `attention_pooling`: Whether to use attention pooling (true/false)
- `stochastic_depth`: Stochastic depth probability (0.0 to 1.0)

### Training Parameters

- `batch_size`: Batch size for training
- `epochs`: Maximum number of training epochs
- `learning_rate`: Initial learning rate
- `min_lr`: Minimum learning rate (for schedulers)
- `weight_decay`: Weight decay (L2 penalty)
- `scheduler`: Learning rate scheduler (cosine, plateau, none)
- `early_stopping_patience`: Number of epochs to wait before early stopping

## Recommended Configurations

Here are some recommended configurations for different scenarios:

### For Maximum Accuracy

```yaml
model_size: '169'
attention: 'cbam'
activation: 'mish'
attention_pooling: true
batch_size: 32
epochs: 200
```

### For Balanced Accuracy/Speed

```yaml
model_size: '121'
attention: 'se'
activation: 'swish'
attention_pooling: false
batch_size: 96
```

### For Maximum Speed

```yaml
model_size: '121'
attention: 'none'
activation: 'relu'
attention_pooling: false
batch_size: 128
stochastic_depth: 0.0
```

## Advanced Training Features

The training script includes several advanced features:

- **Early Stopping**: Automatically stops training when validation metrics plateau
- **Mixed Precision Training**: Uses PyTorch AMP for faster training and reduced memory usage
- **Learning Rate Scheduling**: Implements cosine annealing and ReduceLROnPlateau
- **Stochastic Depth**: Randomly drops layers during training for better generalization
- **Checkpoint Management**: Saves the best model and periodic checkpoints

## Using Checkpoints

Model checkpoints are saved to the `checkpoints/` directory:
- Best model: `densenet_{model_size}_best.pth`
- Regular checkpoints: `densenet_{model_size}_epoch_{epoch}.pth`

To load a checkpoint for inference or continued training:

```python
import torch
from models.densenet import DenseNet121

# Load checkpoint
checkpoint = torch.load('checkpoints/densenet_121_best.pth')

# Create model with the same configuration
model = DenseNet121(
    num_classes=1000,  # Match the number of classes in your dataset
    use_attention="cbam",
    activation="mish",
    use_attention_pooling=True,
    stochastic_depth_prob=0.1
)

# Load the state dict
model.load_state_dict(checkpoint['model_state_dict'])

# For inference mode
model.eval()
```

## Directory Structure

- `models/densenet.py`: DenseNet model implementation with attention mechanisms
- `train_densenet.py`: Training script with advanced training techniques
- `main.py`: Main script that loads configuration and starts training
- `configs/densenet.yaml`: Configuration file for DenseNet training
- `checkpoints/`: Directory for saving model checkpoints
- `logs/`: Directory for saving training logs








