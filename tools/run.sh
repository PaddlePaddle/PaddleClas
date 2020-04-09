#!/usr/bin/env bash

export PYTHONPATH=$(dirname "$PWD"):$PWD:$PYTHONPATH

#python download.py -a ResNet181 -p ./pretrained/ -d 1

#python download.py -a ResNet18 -p ./pretrained/ -d 1

#python download.py -a ResNet34 -p ./pretrained/ -d 0

#python -m paddle.distributed.launch --selected_gpus="0,1,2,3" --log_dir=mylog tools/train.py

#python -m paddle.distributed.launch --selected_gpus="0,1,2,3" --log_dir=mylog ./eval.py

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    --log_dir=mylog \
    tools/train.py \
        -c configs/ResNet/ResNet50_vd.yaml \
        -o use_mix=0 \
        -o TRAIN.batch_size=128 \
        -o TRAIN.transforms.3.NormalizeImage.mean.2=0.4
