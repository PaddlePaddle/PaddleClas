#!/usr/bin/env bash

export PYTHONPATH=$PWD:$PYTHONPATH

python tools/download.py -a ResNet34 -p ./pretrained/ -d 1
