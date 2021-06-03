#!/usr/bin/env bash

python3.7 -m paddle.distributed.launch \
    --gpus="0" \
    tools/train_logo.py \
        -c ./ppcls/configs/Logo/ResNet50.yaml \
        -o print_interval=10 
