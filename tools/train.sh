#!/usr/bin/env bash

# for single card train
# python tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards train
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml
