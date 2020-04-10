#!/usr/bin/env bash

export PYTHONPATH=$PWD:$PYTHONPATH

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    --log_dir=log_ResNet50 \
    tools/train.py \
        -c ./configs/ResNet/ResNet50.yaml
