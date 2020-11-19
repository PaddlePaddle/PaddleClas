#!/usr/bin/env bash

export FLAGS_fraction_of_gpu_memory_to_use=0.80

python3.7 -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c configs/ResNet/ResNet50.yaml \
        -o TRAIN.batch_size=256 \
        -o use_dali=True
