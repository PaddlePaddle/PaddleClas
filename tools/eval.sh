#!/usr/bin/env bash

# for single card eval
# python3.7 tools/eval.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards eval
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/eval.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml -o Global.use_dali=True -o Global.pretrained_model=./ResNet50_pretrained
