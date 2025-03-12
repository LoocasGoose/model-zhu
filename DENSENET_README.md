# DenseNet for Medium ImageNet

This implementation contains the DenseNet architecture for image classification, based on the paper "Densely Connected Convolutional Networks" by Huang et al. (https://arxiv.org/abs/1608.06993).

## Architecture

DenseNet features dense connectivity between layers, where each layer receives inputs from all preceding layers and passes its feature maps to all subsequent layers. This connectivity pattern has several advantages:

1. Alleviates the vanishing gradient problem
2. Strengthens feature propagation
3. Encourages feature reuse
4. Substantially reduces the number of parameters

## Implemented Models

This implementation includes three variants of DenseNet:

1. **DenseNet-121**: 4 dense blocks with [6, 12, 24, 16] layers
2. **DenseNet-169**: 4 dense blocks with [6, 12, 32, 32] layers  
3. **DenseNet-201**: 4 dense blocks with [6, 12, 48, 32] layers

## Key Components

- **Bottleneck Layers**: Each layer in a dense block consists of a BN-ReLU-Conv(1×1) followed by BN-ReLU-Conv(3×3)
- **Growth Rate**: Controls how many new features each layer contributes to the global state
- **Transition Layers**: Placed between dense blocks to reduce feature map size via convolution and pooling
- **Compression**: Transition layers can compress the number of channels to improve model efficiency

## Training

To train the DenseNet-121 model on Medium ImageNet, run:

```
python main.py --cfg configs/densenet121_medium_imagenet.yaml
```

On the honeydew server:

```
CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/densenet121_medium_imagenet.yaml
```

## Configuration

The configuration file `configs/densenet121_medium_imagenet.yaml` contains hyperparameters for training DenseNet-121 on Medium ImageNet:

- **Batch Size**: 64
- **Learning Rate**: 3e-4 with cosine decay
- **Optimizer**: AdamW
- **Data Augmentation**: Color jitter with magnitude 0.4
- **Input Size**: 32×32
- **Small Inputs**: True (uses smaller stem design for CIFAR-sized inputs)
- **Dataset Path**: By default, the model will use the Medium ImageNet dataset at `/data/imagenet/medium-imagenet-nmep-96.hdf5`. If needed, you can specify a different path in the config:
  ```yaml
  DATA:
    MEDIUM_IMAGENET_PATH: "/path/to/medium-imagenet.hdf5"
  ```

## Performance

DenseNet generally achieves better performance compared to other architectures with a similar number of parameters due to its improved feature reuse. On ImageNet, the original paper reported that DenseNet-201 achieved comparable accuracy to ResNet-101 while using fewer parameters. 