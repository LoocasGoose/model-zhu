#!/bin/bash
echo "Training on 10% of the full dataset"
python main.py --cfg configs/resnet18_base.yaml --subset-fraction 0.1 --opts DATA.BATCH_SIZE 256 DATA.NUM_WORKERS 32 TRAIN.EPOCHS 20 