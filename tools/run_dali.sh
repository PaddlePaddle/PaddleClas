#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export FLAGS_fraction_of_gpu_memory_to_use=0.80

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3,4,5,6,7" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50.yaml \
        -o print_interval=10 \
        -o use_dali=true
