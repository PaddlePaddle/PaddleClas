# 图像分类昆仑模型介绍(持续更新中)

## 前言

* 本文档介绍了目前昆仑支持的模型以及如何在昆仑设备上训练这些模型。支持昆仑的PaddlePaddle安装参考install_kunlun(https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/install/install_Kunlun_zh.md)

## 昆仑训练
* 数据来源和预训练模型参考[quick_start](../quick_start/quick_start_classification_new_user.md)。昆仑训练效果与CPU/GPU对齐。

### ResNet50
* 命令：

```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/ResNet50_vd_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```

与cpu/gpu训练的区别是加上-o use_xpu=True, 表示执行在昆仑设备上。

### MobileNetV3
* 命令：

```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```

### HRNet
* 命令：

```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/kunlun/HRNet_W18_C_finetune_kunlun.yaml \
    -o is_distributed=False \
    -o use_xpu=True \
    -o use_gpu=False
```


### VGG16/19
* 命令：

```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/VGG16_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```
```shell
python3.7 ppcls/static/train.py \
    -c ppcls/configs/quick_start/VGG19_finetune_kunlun.yaml \
    -o use_gpu=False \
    -o use_xpu=True \
    -o is_distributed=False
```
