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
CUDA_VISIBLE_DEVICES=0 python main.py --cfg=configs/lenet_base.yaml
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

