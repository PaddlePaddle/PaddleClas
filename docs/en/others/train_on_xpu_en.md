# Introduction to Image Classification Model Kunlun (Continuously updated)

------

## Catalogue

- [1. Foreword](#1)
- [2. Training of Kunlun](#2)
  - [2.1 ResNet50](#2.1)
  - [2.2 MobileNetV3](#2.2)
  - [2.3 HRNet](#2.3)
  - [2.4 VGG16/19](#2.4)

<a name='1'></a>

## 1. Forword

- This document describes the models currently supported by Kunlun and how to train these models on Kunlun devices. To install PaddlePaddle that supports Kunlun, please refer to [install_kunlun](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/09_hardware_support/xpu_docs/paddle_install_cn.html)

<a name='2'></a>

## 2. Training of Kunlun

- See [quick_start](../quick_start/quick_start_classification_new_user_en.md)for data sources and pre-trained models. The training effect of Kunlun is aligned with CPU/GPU.

<a name='2.1'></a>

### 2.1 ResNet50

- Command:

```
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/ResNet50_vd_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```

The difference with cpu/gpu training lies in the addition of -o use_xpu=True, indicating that the execution is on a Kunlun device.

<a name='2.2'></a>

### 2.2 MobileNetV3

- Command：

```
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```

<a name='2.3'></a>

### 2.3 HRNet

- Command：

```
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/HRNet_W18_C_finetune_kunlun.yaml \
    -o is_distributed=False \
    -o use_xpu=True \
    -o use_gpu=False
```

<a name='2.4'></a>

### 2.4 VGG16/19

- Command：

```
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/VGG16_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/VGG19_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```
