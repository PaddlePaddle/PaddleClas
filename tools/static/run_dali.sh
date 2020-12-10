#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export FLAGS_fraction_of_gpu_memory_to_use=0.80

python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/static/train.py \
        -c ./configs/ResNet/ResNet50.yaml \
        -o print_interval=10 \
        -o use_dali=True
