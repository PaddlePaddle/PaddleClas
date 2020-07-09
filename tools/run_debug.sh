#!/usr/bin/env bash

export PYTHONPATH=$PWD:$PYTHONPATH

python -m paddle.distributed.launch \
    --selected_gpus="0" \
    tools/train.py \
        -c ./configs/debug.yaml 
