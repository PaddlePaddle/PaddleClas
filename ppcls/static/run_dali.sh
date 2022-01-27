#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    ppcls/static/train.py \
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50_amp_O1.yaml
