#!/bin/bash
python main.py --cfg configs/resnet18_base.yaml --subset-fraction 1.0 --opts DATA.BATCH_SIZE 2048 DATA.NUM_WORKERS 8 TRAIN.EPOCHS 50 