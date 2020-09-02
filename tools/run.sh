#!/usr/bin/env bash

export PYTHONPATH=$PWD:$PYTHONPATH

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50_vd.yaml \
        -o print_interval=10
