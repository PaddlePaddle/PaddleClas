# Introduction to Image Classification Model Kunlun (Continuously updated)

------

## Contents

- [1. Foreword](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.3/docs/zh_CN/others/train_on_xpu.md#1)
- [2. Training of Kunlun](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.3/docs/zh_CN/others/train_on_xpu.md#2)
  - [2.1 ResNet50](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.3/docs/zh_CN/others/train_on_xpu.md#2.1)
  - [2.2 MobileNetV3](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.3/docs/zh_CN/others/train_on_xpu.md#2.2)
  - [2.3 HRNet](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.3/docs/zh_CN/others/train_on_xpu.md#2.3)
  - [2.4 VGG16/19](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.3/docs/zh_CN/others/train_on_xpu.md#2.4)



## 1. Forword

- This document describes the models currently supported by Kunlun and how to train these models on Kunlun devices. To install PaddlePaddle that supports Kunlun, please refer to install_kunlun(https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/install/install_Kunlun_zh.md)



## 2. Training of Kunlun

- See [quick_start](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/quick_start/quick_start_ classification_new_user.md) for data sources and pre-trained models. The training effect of Kunlun is aligned with CPU/GPU.



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



### 2.2 MobileNetV3

- Command：

```
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```



### 2.3 HRNet

- Command：

```
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/HRNet_W18_C_finetune_kunlun.yaml \
    -o is_distributed=False \
    -o use_xpu=True \
    -o use_gpu=False
```



### 2.4 VGG16/19

- Command：

```
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/VGG16_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/VGG19_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```
