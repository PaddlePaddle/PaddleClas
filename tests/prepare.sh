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
    if [ ${#array[*]} = 2 ]; then
        echo ${array[1]}
    else
    	IFS="|"
    	tmp="${array[1]}:${array[2]}"
        echo ${tmp}
    fi
}
model_name=$(func_parser_value "${lines[1]}")
inference_model_url=$(func_parser_value "${lines[35]}")

if [ ${MODE} = "lite_train_infer" ] || [ ${MODE} = "whole_infer" ];then
    # pretrain lite train data
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_little_train.tar
    tar xf whole_chain_little_train.tar
    ln -s whole_chain_little_train ILSVRC2012
    cd ILSVRC2012 
    mv train.txt train_list.txt
    mv val.txt val_list.txt
    cd ../../
elif [ ${MODE} = "infer" ];then
    # download data
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_infer.tar
    tar xf whole_chain_infer.tar
    ln -s whole_chain_infer ILSVRC2012
    cd ILSVRC2012 
    mv val.txt val_list.txt
    cd ../../
    # download inference model
    eval "wget -nc $inference_model_url"
    tar xf "${model_name}_inference.tar"

elif [ ${MODE} = "whole_train_infer" ];then
    cd dataset
    rm -rf ILSVRC2012
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_CIFAR100.tar
    tar xf whole_chain_CIFAR100.tar
    ln -s whole_chain_CIFAR100 ILSVRC2012
    cd ILSVRC2012 
    mv train.txt train_list.txt
    mv val.txt val_list.txt
    cd ../../
fi
