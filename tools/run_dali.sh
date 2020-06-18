#!/usr/bin/env bash

export PYTHONPATH=$PWD:$PYTHONPA
export FLAGS_fraction_of_gpu_memory_to_use=0.8


python3 -m paddle.distributed.launch \
    --selected_gpus="0,1" \
    tools/train.py \
        -c ./configs/high_performance/dali.yaml \
        -o TRAIN.batch_size=4 \
	-o use_mix=True \
        -o use_dali=True
