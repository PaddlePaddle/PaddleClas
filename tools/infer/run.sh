#!/usr/bin/env bash

python ./cpp_infer.py \
    -i=./test.jpeg \
    -m=./resnet50-vd/model \
    -p=./resnet50-vd/params \
    --use_gpu=1

python ./cpp_infer.py \
    -i=./test.jpeg \
    -m=./resnet50-vd/model \
    -p=./resnet50-vd/params \
    --use_gpu=0

python py_infer.py \
    -i=./test.jpeg \
    -d ./resnet50-vd/ \
    -m=model -p=params \
    --use_gpu=0

python py_infer.py \
    -i=./test.jpeg \
    -d ./resnet50-vd/ \
    -m=model -p=params \
    --use_gpu=1

python infer.py \
    -i=./test.jpeg \
    -m ResNet50_vd \
    -p ./resnet50-vd-persistable/ \
    --use_gpu=0

python infer.py \
    -i=./test.jpeg \
    -m ResNet50_vd \
    -p ./resnet50-vd-persistable/ \
    --use_gpu=1

python export_model.py \
    -m ResNet50_vd \
    -p ./resnet50-vd-persistable/ \
    -o ./test/

python py_infer.py \
    -i=./test.jpeg \
    -d ./test/ \
    -m=model \
    -p=params \
    --use_gpu=0
