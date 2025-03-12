# HW 1 Worksheet

---

This is the worksheet for Homework 1. Your deliverables for this homework are:

- [ ] This worksheet with all answers filled in. If you include plots/images, be sure to include all the required files. Alternatively, you can export it as a PDF and it will be self-sufficient.
- [ ] Kaggle submission and writeup (details below)
- [ ] Github repo with all of your code! You need to either fork it or just copy the code over to your repo. A simple way of doing this is provided below. Include the link to your repo below. If you would like to make the repo private, please dm us and we'll send you the GitHub usernames to add as collaborators.

`YOUR GITHUB REPO HERE (or notice that you DMed us to share a private repo)`

## To move to your own repo:

First follow `README.md` to clone the code. Additionally, create an empty repo on GitHub that you want to use for your code. Then run the following commands:

```bash
$ git remote rename origin staff # staff is now the provided repo
$ git remote add origin <your repos remote url>
$ git push -u origin main
```



# Part -1: PyTorch review

Feel free to ask your NMEP friends if you don't know!

## -1.0 What is the difference between `torch.nn.Module` and `torch.nn.functional`?

`torch.nn.Module` is a base class for neural network layers/blocks with learnable parameters. `torch.nn.functional` contains stateless functions for operations like activation functions and loss calculations.

## -1.1 What is the difference between a Dataset and a DataLoader?

A Dataset is a class that contains the data and labels. A DataLoader is a class that wraps a Dataset and provides a way to iterate over the data in batches.

## -1.2 What does `@torch.no_grad()` above a function header do?

`@torch.no_grad()` disables gradient tracking during inference to reduce memory usage and speed up computations.



# Part 0: Understanding the codebase

Read through `README.md` and follow the steps to understand how the repo is structured.

## 0.0 What are the `build.py` files? Why do we have them?**

`build.py` files are used to build the model and data loader using the configs. They are used to separate the config parsing from the model and data loader definitions.

## 0.1 Where would you define a new model?

`models/build.py`

## 0.2 How would you add support for a new dataset? What files would you need to change?

`data/build.py` and `data/datasets.py` would need to be changed.

## 0.3 Where is the actual training code?

`main.py`

## 0.4 Create a diagram explaining the structure of `main.py` and the entire code repo.

Be sure to include the 4 main functions in it (`main`, `train_one_epoch`, `validate`, `evaluate`) and how they interact with each other. Also explain where the other files are used. No need to dive too deep into any part of the code for now, the following parts will do deeper dives into each part of the code. For now, read the code just enough to understand how the pieces come together, not necessarily the specifics. You can use any tool to create the diagram (e.g. just explain it in nice markdown, draw it on paper and take a picture, use draw.io, excalidraw, etc.)

`main.py` is the main training loop. It defines the training loop, validation loop, and evaluation loop. It also handles the loading of the model, data loader, and optimizer.



# Part 1: Datasets

The following questions relate to `data/build.py` and `data/datasets.py`.

## 1.0 Builder/General

### 1.0.0 What does `build_loader` do?

`build_loader` is a function that builds the data loader using the config. It takes in a dataset class and returns a data loader.

### 1.0.1 What functions do you need to implement for a PyTorch Datset? (hint there are 3)

`__getitem__`, `__len__`, and `__init__` are the three functions that need to be implemented for a PyTorch Dataset.

## 1.1 CIFAR10Dataset

### 1.1.0 Go through the constructor. What field actually contains the data? Do we need to download it ahead of time?

`self.data` contains the data. We need to download it ahead of time.

### 1.1.1 What is `self.train`? What is `self.transform`?

`self.train` is a boolean that indicates if the dataset is for training or not. `self.transform` is a function that transforms the data.

### 1.1.2 What does `__getitem__` do? What is `index`?

`__getitem__` is a function that returns the item at index `index`.

### 1.1.3 What does `__len__` do?

`__len__` is a function that returns the length of the dataset.

### 1.1.4 What does `self._get_transforms` do? Why is there an if statement?

`self._get_transforms` is a function that returns the transforms for the dataset. The if statement is used to check if the dataset is for training or not.

### 1.1.5 What does `transforms.Normalize` do? What do the parameters mean? (hint: take a look here: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

`transforms.Normalize` is a function that normalizes the data. The parameters are the mean and standard deviation of the data.

## 1.2 MediumImagenetHDF5Dataset

### 1.2.0 Go through the constructor. What field actually contains the data? Where is the data actually stored on honeydew? What other files are stored in that folder on honeydew? How large are they?

`self.file` contains the data. The data is stored on honeydew at `/data/medium-imagenet/medium-imagenet-nmep-96.hdf5`. The other files are stored in that folder on honeydew. This folder contains HDF5 files for different image sizes (96x96 and 224x224) and splits (train, val, test), as well as metadata files like class names. The files are quite large - the full dataset is 1.5M training images plus 190k images each for validation and test.

> *Some background*: HDF5 is a file format that stores data in a hierarchical structure. It is similar to a python dictionary. The files are binary and are generally really efficient to use. Additionally, `h5py.File()` does not actually read the entire file contents into memory. Instead, it only reads the data when you access it (as in `__getitem__`). You can learn more about [hdf5 here](https://portal.hdfgroup.org/display/HDF5/HDF5) and [h5py here](https://www.h5py.org/).

### 1.2.1 How is `_get_transforms` different from the one in CIFAR10Dataset?

The _get_transforms in MediumImagenetHDF5Dataset uses different normalization values (mean and std) specific to the Medium ImageNet dataset. 

### 1.2.2 How is `__getitem__` different from the one in CIFAR10Dataset? How many data splits do we have now? Is it different from CIFAR10? Do we have labels/annotations for the test set?

The __getitem__ in MediumImagenetHDF5Dataset reads data from HDF5 files rather than from memory. It has three data splits: train, val, and test (whereas CIFAR10 just has train and test). Unlike CIFAR10, the test set labels are not provided for Medium ImageNet to prevent cheating in the competition - the implementation would return -1 or None for test labels.

### 1.2.3 Visualizing the dataset

Visualize ~10 or so examples from the dataset. There's many ways to do it - you can make a separate little script that loads the datasets and displays some images, or you can update the existing code to display the images where it's already easy to load them. In either case, you can use use `matplotlib` or `PIL` or `opencv` to display/save the images. Alternatively you can also use `torchvision.utils.make_grid` to display multiple images at once and use `torchvision.utils.save_image` to save the images to disk.

Be sure to also get the class names. You might notice that we don't have them loaded anywhere in the repo - feel free to fix it or just hack it together for now, the class names are in a file in the same folder as the hdf5 dataset.

`YOUR ANSWER HERE`


# Part 2: Models

The following questions relate to `models/build.py` and `models/models.py`.

## What models are implemented for you?

LeNet is implemented as a baseline model. It is a classic convolutional neural network architecture from the 1990s designed by Yann LeCun. The implementation serves as a template for implementing more complex architectures like AlexNet and ResNet.

## What do PyTorch models inherit from? What functions do we need to implement for a PyTorch Model? (hint there are 2)

PyTorch models inherit from `torch.nn.Module`. The two functions we need to implement are:
1. `__init__`: Constructor that initializes the model's layers and parameters
2. `forward`: Defines how data flows through the model (the forward pass)

## How many layers does our implementation of LeNet have? How many parameters does it have? (hint: to count the number of parameters, you might want to run the code)

The LeNet implementation has 7 layers:
- 2 convolutional layers:
  - Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
  - Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
- 2 pooling layers:
  - AvgPool2d(kernel_size=2, stride=2) (×2)
- 3 fully connected layers:
  - Linear(16 * 6 * 6, 120)
  - Linear(120, 84)
  - Linear(84, num_classes)

The total parameter count is approximately 61,706 parameters, with most parameters in the first fully connected layer (16 * 6 * 6 * 120 = 69,120 parameters).



# Part 3: Training

The following questions relate to `main.py`, and the configs in `configs/`.

## 3.0 What configs have we provided for you? What models and datasets do they train on?

The configs provided include:
- `lenet_base.yaml`: LeNet on CIFAR10 with batch size 256, input size 32×32, using AdamW optimizer
- `resnet18_base.yaml`: ResNet18 on CIFAR10 with batch size 1024, input size 32×32
- `resnet18_medium_imagenet.yaml`: ResNet18 on Medium ImageNet with batch size 32, input size 32×32

These configs specify hyperparameters including learning rates (3e-4), batch sizes, optimizers (AdamW), training schedules (cosine learning rate decay), and number of epochs.

## 3.1 Open `main.py` and go through `main()`. In bullet points, explain what the function does.

The `main()` function in `main.py`:
- Parses command line arguments and loads configuration from YAML file
- Sets up output directory for logs and checkpoints
- Creates logger for tracking training progress
- Sets random seeds for reproducibility
- Builds the model according to config specifications
- Moves model to GPU if available
- Creates data loaders for training, validation, and test sets
- Sets up the optimizer (AdamW by default) and learning rate scheduler (cosine decay)
- Loads checkpoint if resuming training
- For each epoch:
  - Calls train_one_epoch() for training
  - Adjusts learning rate according to schedule
  - Calls validate() to check performance on validation set
  - Saves checkpoint if validation improves or at specified intervals
- Calls evaluate() on the test set and creates submission file
- Returns best validation accuracy

## 3.2 Go through `validate()` and `evaluate()`. What do they do? How are they different? 
> Could we have done better by reusing code? Yes. Yes we could have but we didn't... sorry...

Both functions put the model in evaluation mode, but serve different purposes:

`validate()`:
- Runs during training to check performance on validation data
- Computes both loss and accuracy metrics on validation set
- Has gradient computation disabled with @torch.no_grad()
- Returns accuracy as a percentage for monitoring training progress
- Results are used to determine if the model should be saved as "best"

`evaluate()`:
- Runs after training is complete to generate predictions on test data
- Only computes model outputs (no loss or accuracy since test labels may not be available)
- Also has gradient computation disabled
- Returns raw model predictions as numpy arrays
- Used primarily to create submission files for competitions
- Doesn't compute metrics or return performance measures

The key difference is that validate() measures performance to guide training, while evaluate() generates predictions for external evaluation/submission.


# Part 4: AlexNet

## Implement AlexNet. Feel free to use the provided LeNet as a template. For convenience, here are the parameters for AlexNet:

```
Input NxNx3 # For CIFAR 10, you can set img_size to 70
Conv 11x11, 64 filters, stride 4, padding 2
MaxPool 3x3, stride 2
Conv 5x5, 192 filters, padding 2
MaxPool 3x3, stride 2
Conv 3x3, 384 filters, padding 1
Conv 3x3, 256 filters, padding 1
Conv 3x3, 256 filters, padding 1
MaxPool 3x3, stride 2
nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
flatten into a vector of length x # what is x?
Dropout 0.5
Linear with 4096 output units
Dropout 0.5
Linear with 4096 output units
Linear with num_classes output units
```

> ReLU activation after every Conv and Linear layer. DO **NOT** Forget to add activatioons after every layer. Do not apply activation after the last layer.

## 4.1 How many parameters does AlexNet have? How does it compare to LeNet? With the same batch size, how much memory do LeNet and AlexNet take up while training? 
> (hint: use `gpuststat`)

lenet has around 60,000, alexnet has around 60,000,000. 
lenet used 573MB, alexnet used 1.9GB.


## 4.2 Train AlexNet on CIFAR10. What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again.
> Also no need to tune the models much, you'll do it in the next part.

[2025-03-12 07:34:06 alexnet](main.py 153): INFO Train: [12/50] lr 0.000259     time 0.0195 (0.0450)    loss 0.3942 (0.3810)  Acc@1 88.750 (86.768)    Mem 1586MB
[2025-03-12 07:34:06 alexnet](main.py 162): INFO EPOCH 12 training takes 0:00:08
[2025-03-12 07:34:06 alexnet](main.py 91): INFO  * Train Acc 86.768 Train Loss 0.381
[2025-03-12 07:34:06 alexnet](main.py 92): INFO Accuracy of the network on the 50000 train images: 86.8%
[2025-03-12 07:34:08 alexnet](main.py 195): INFO Validate:      Time 0.003 (0.040)      Loss 0.5816 (0.5875)    Acc@1 87.500 (80.550)  Mem 1586MB
[2025-03-12 07:34:08 alexnet](main.py 96): INFO  * Val Acc 80.550 Val Loss 0.588
[2025-03-12 07:34:08 alexnet](main.py 97): INFO Accuracy of the network on the 10000 val images: 80.5%
[2025-03-12 07:34:08 alexnet](main.py 103): INFO Max accuracy: 80.55%



# Part 5: Weights and Biases

> Parts 5 and 6 are independent. Feel free to attempt them in any order you want.

> Background on W&B. W&B is a tool for tracking experiments. You can set up experiments and track metrics, hyperparameters, and even images. It's really neat and we highly recommend it. You can learn more about it [here](https://wandb.ai/site).
> 
> For this HW you have to use W&B. The next couple parts should be fairly easy if you setup logging for configs (hyperparameters) and for loss/accuracy. For a quick tutorial on how to use it, check out [this quickstart](https://docs.wandb.ai/quickstart). We will also cover it at HW party at some point this week if you need help.

## 5.0 Setup plotting for training and validation accuracy and loss curves. Plot a point every epoch.

`PUSH YOUR CODE TO YOUR OWN GITHUB :)`

## 5.1 Plot the training and validation accuracy and loss curves for AlexNet and LeNet. Attach the plot and any observations you have below.

`YOUR ANSWER HERE`

## 5.2 For just AlexNet, vary the learning rate by factors of 3ish or 10 (ie if it's 3e-4 also try 1e-4, 1e-3, 3e-3, etc) and plot all the loss plots on the same graph. What do you observe? What is the best learning rate? Try at least 4 different learning rates.

`YOUR ANSWER HERE`

## 5.3 Do the same with batch size, keeping learning rate and everything else fixed. Ideally the batch size should be a power of 2, but try some odd batch sizes as well. What do you observe? Record training times and loss/accuracy plots for each batch size (should be easy with W&B). Try at least 4 different batch sizes.

`YOUR ANSWER HERE`

## 5.4 As a followup to the previous question, we're going to explore the effect of batch size on _throughput_, which is the number of images/sec that our model can process. You can find this by taking the batch size and dividing by the time per epoch. Plot the throughput for batch sizes of powers of 2, i.e. 1, 2, 4, ..., until you reach CUDA OOM. What is the largest batch size you can support? What trends do you observe, and why might this be the case?
You only need to observe the training for ~ 5 epochs to average out the noise in training times; don't train to completion for this question! We're only asking about the time taken. If you're curious for a more in-depth explanation, feel free to read [this intro](https://horace.io/brrr_intro.html). 

`YOUR ANSWER HERE`

## 5.5 Try different data augmentations. Take a look [here](https://pytorch.org/vision/stable/transforms.html) for torchvision augmentations. Try at least 2 new augmentation schemes. Record loss/accuracy curves and best accuracies on validation/train set.

`YOUR ANSWER HERE`

## 5.6 (optional) Play around with more hyperparameters. I recommend playing around with the optimizer (Adam, SGD, RMSProp, etc), learning rate scheduler (constant, StepLR, ReduceLROnPlateau, etc), weight decay, dropout, activation functions (ReLU, Leaky ReLU, GELU, Swish, etc), etc.

`YOUR ANSWER HERE`



# Part 6: ResNet

## 6.0 Implement and train ResNet18

In `models/*`, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

`YOUR ANSWER HERE`

## 6.1 Visualize examples

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy.

`YOUR ANSWER HERE`


# Part 7: Kaggle submission

To make this more fun, we have scraped an entire new dataset for you! 🎉

We called it MediumImageNet. It contains 1.5M training images, and 190k images for validation and test each. There are 200 classes distributed approximately evenly. The images are available in 224x224 and 96x96 in hdf5 files. The test set labels are not provided :). 

The dataset is downloaded onto honeydew at `/data/medium-imagenet`. Feel free to play around with the files and learn more about the dataset.

For the kaggle competition, you need to train on the 1.5M training images and submit predictions on the 190k test images. You may validate on the validation set but you may not use is as a training set to get better accuracy (aka don't backprop on it). The test set labels are not provided. You can submit up to 10 times a day (hint: that's a lot). The competition ends on __TBD__.

Your Kaggle scores should approximately match your validation scores. If they do not, something is wrong.

(Soon) when you run the training script, it will output a file called `submission.csv`. This is the file you need to submit to Kaggle. You're required to submit at least once. 

## Kaggle writeup

We don't expect anything fancy here. Just a brief summary of what you did, what worked, what didn't, and what you learned. If you want to include any plots, feel free to do so. That's brownie points. Feel free to write it below or attach it in a separate file.

**REQUIREMENT**: Everyone in your group must be able to explain what you did! Even if one person carries (I know, it happens) everyone must still be able to explain what's going on!

Now go play with the models and have some competitive fun! 🎉
