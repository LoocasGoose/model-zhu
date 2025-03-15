@echo off
echo Training ResNet50V2 with anti-overfitting configuration (batch size 64)

python train_resnetv2.py ^
  --cfg configs/resnet50v2.yaml ^
  --model-variant resnet50v2 ^
  --output output/resnet50v2_antioverfitting_bs64 ^
  --workers 4 ^
  --batch-size 64 ^
  --opts ^
  MODEL.DROP_RATE 0.3 ^
  MODEL.STOCHASTIC_DEPTH_RATE 0.2 ^
  TRAIN.LR 0.005 ^
  TRAIN.WARMUP_EPOCHS 10 ^
  TRAIN.OPTIMIZER.WEIGHT_DECAY 0.0001 ^
  TRAIN.OPTIMIZER.NESTEROV True ^
  TRAIN.GRADIENT_ACCUMULATION_STEPS 4 ^
  TRAIN.GRADIENT_CLIP_VAL 1.0 ^
  AUG.MIXUP 0.4 ^
  AUG.CUTMIX 0.4 ^
  AUG.RANDOM_ERASING 0.25 ^
  AUG.LABEL_SMOOTHING 0.1 ^
  MEMORY.EMPTY_CACHE_FREQ 20

echo Training complete! 