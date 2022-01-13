# 图像分类昆仑模型介绍(持续更新中)
------
## 目录
* [1. 前言](#1)
* [2. 昆仑训练](#2)
	* [2.1 ResNet50](#2.1)
	* [2.2 MobileNetV3](#2.2)
	* [2.3 HRNet](#2.3)
	* [2.4 VGG16/19](#2.4)

 <a name='1'></a>

## 1. 前言

* 本文档介绍了目前昆仑支持的模型以及如何在昆仑设备上训练这些模型。支持昆仑的 PaddlePaddle 安装参考 install_kunlun(https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/install/install_Kunlun_zh.md)

<a name='2'></a>

## 2. 昆仑训练
* 数据来源和预训练模型参考[quick_start](../quick_start/quick_start_classification_new_user.md)。昆仑训练效果与 CPU/GPU 对齐。

<a name='2.1'></a>

### 2.1 ResNet50
* 命令：

```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/ResNet50_vd_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```

与 cpu/gpu 训练的区别是加上 -o use_xpu=True, 表示执行在昆仑设备上。

 <a name='2.2'></a>

### 2.2 MobileNetV3
* 命令：

```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```

<a name='2.3'></a>

### 2.3 HRNet
* 命令：

```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/HRNet_W18_C_finetune_kunlun.yaml \
    -o is_distributed=False \
    -o use_xpu=True \
    -o use_gpu=False
```

<a name='2.4'></a>

### 2.4 VGG16/19
* 命令：

```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/VGG16_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```
```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/VGG19_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```
