# Getting Started
---
Please refer to [Installation](install.md) to setup environment at first, and prepare ImageNet1K data by following the instruction mentioned in the [data](data.md)

## Setup

**Setup PYTHONPATH：**

```bash
export PYTHONPATH=path_to_PaddleClas:$PYTHONPATH
```

## Training and validating

PaddleClas support `tools/train.py` and `tools/eval.py` to start training and validating.

### Training

```bash
# PaddleClas use paddle.distributed.launch to start multi-cards and multiprocess training.
# Set FLAGS_selected_gpus to indicate GPU cards

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50_vd.yaml
```

- log:

```
epoch:0    train    step:13    loss:7.9561    top1:0.0156    top5:0.1094    lr:0.100000    elapse:0.193
```

add -o params to update configuration

```bash
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50_vd.yaml \
        -o use_mix=1 \
    --vdl_dir=./scalar/

```

- log:

```
epoch:0    train    step:522    loss:1.6330    lr:0.100000    elapse:0.210
```

or modify configuration directly to config fileds, please refer to [config](config.md) for more details.

use visuldl to visulize training loss in the real time

```bash
visualdl --logdir ./scalar --host <host_IP> --port <port_num>

```


### finetune

* please refer to [Trial](./quick_start.md) for more details.

### validating

```bash
python tools/eval.py \
    -c ./configs/eval.yaml \
    -o ARCHITECTURE.name="ResNet50_vd" \
    -o pretrained_model=path_to_pretrained_models

modify `configs/eval.yaml filed: `ARCHITECTURE.name` and filed: `pretrained_model` to config valid model or add -o params to update config directly.


**NOTE: ** when loading the pretrained model, should ignore the suffix ```.pdparams```

## Predict

PaddlePaddle supprot three predict interfaces
Use predicator interface to predict
First, export inference model

```bash
python tools/export_model.py \
    --model=model_name \
    --pretrained_model=pretrained_model_dir \
    --output_path=save_inference_dir

```
Second, start predicator enginee：

```bash
python tools/infer/predict.py \
    -m model_path \
    -p params_path \
    -i image path \
    --use_gpu=1 \
    --use_tensorrt=True
```
please refer to [inference](../extension/paddle_inference.md) for more details.
