@echo off
echo Training ResNet18V2 with anti-overfitting configuration

python train_resnetv2.py ^
  --cfg configs/resnet18v2.yaml ^
  --model-variant resnet18v2 ^
  --output output/resnet18v2_antioverfitting ^
  --workers 4 ^
  --batch-size 128 ^
  --opts ^
  MODEL.DROP_RATE 0.3 ^
  MODEL.STOCHASTIC_DEPTH_RATE 0.2 ^
  TRAIN.WARMUP_EPOCHS 10 ^
  TRAIN.OPTIMIZER.WEIGHT_DECAY 0.0001 ^
  AUG.MIXUP 0.4 ^
  AUG.CUTMIX 0.4 ^
  AUG.RANDOM_ERASING 0.25 ^
  AUG.LABEL_SMOOTHING 0.1
  
echo Training complete! 