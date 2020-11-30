#!/usr/bin/env bash

# IP Addresses of all nodes, modify it corresponding to your own environment
ALL_NODE_IPS="10.10.10.1,10.10.10.2"
# IP Address of the current node, modify it corresponding to your own environment
CUR_NODE_IPS="10.10.10.1"

python -m paddle.distributed.launch \
    --cluster_node_ips=$ALL_NODE_IPS \
    --node_ip=$CUR_NODE_IPS \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50.yaml \
        -o print_interval=10
