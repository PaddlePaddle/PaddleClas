#!/usr/bin/env bash

python -m paddle.distributed.launch \
    --gpus="0" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50.yaml \
        -o print_interval=10
