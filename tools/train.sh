#!/usr/bin/env bash

# for single card train
# python3.7 tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards train
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python3.7 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" \
# tools/train.py -c ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml -o Global.use_dali=True -o Dataloader.Train.loader.num_workers=8 -o Global.output_dir="./output/PP-ShiTuV2_DALI"
python3.7 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" \
tools/train.py -c ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml -o Global.use_dali=False -o Dataloader.Train.loader.num_workers=8 -o Global.output_dir="./output/PP-ShiTuV2_debug"

# python3.7 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" \
# tools/train.py -c ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml -o Global.use_dali=False -o Dataloader.Train.loader.num_workers=8 -o Global.output_dir="output/PP-ShiTuV2_wo_DALI"
