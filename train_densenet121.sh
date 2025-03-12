#!/bin/bash
# Train DenseNet121 on Medium ImageNet dataset located at /data/imagenet
CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/densenet121_medium_imagenet.yaml

