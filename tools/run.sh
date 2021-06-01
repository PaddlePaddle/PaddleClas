#!/usr/bin/env bash

python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
        -o print_interval=10
