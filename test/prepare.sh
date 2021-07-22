#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2

dataline=$(cat ${FILENAME})
# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    IFS="|"
    tmp="${array[1]}:${array[2]}"
    echo ${tmp}
}
ResNet50_vd=$(func_parser_value "${lines[49]}")
ResNeXt101_vd_64x4d=$(func_parser_value "${lines[50]}")
HRNet_W18_C=$(func_parser_value "${lines[51]}")
MobileNetV3_large_x1_0=$(func_parser_value "${lines[52]}")
DarkNet53=$(func_parser_value "${lines[53]}")
MobileNetV1=$(func_parser_value "${lines[54]}")
MobileNetV2=$(func_parser_value "${lines[55]}")
ShuffleNetV2_x1_0=$(func_parser_value "${lines[56]}")

if [ ${MODE} = "lite_train_infer" ] || [ ${MODE} = "whole_infer" ];then
    # pretrain lite train data
    cd dataset
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_little_train.tar
    tar xf whole_chain_little_train.tar
    ln -s whole_chain_little_train chain_dataset
    cd ../
elif [ ${MODE} = "infer" ];then
    # download data
    cd dataset
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_infer.tar
    tar xf whole_chain_infer.tar
    ln -s whole_chain_infer chain_dataset
    cd ../
    # download pretrained model
    mkdir -p pretrained_models
    cd pretrained_models
    eval "wget -nc $ResNet50_vd"
    eval "wget -nc $ResNeXt101_vd_64x4d"
    eval "wget -nc $HRNet_W18_C"
    eval "wget -nc $MobileNetV3_large_x1_0"
    eval "wget -nc $DarkNet53"
    eval "wget -nc $MobileNetV1"
    eval "wget -nc $MobileNetV2"
    eval "wget -nc $ShuffleNetV2_x1_0"

elif [ ${MODE} = "whole_train_infer" ];then
    cd dataset
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_CIFAR100.tar
    tar xf whole_chain_CIFAR100.tar
    ln -s whole_chain_CIFAR100 chain_dataset
fi
