# 图像分类昆仑模型介绍(持续更新中)

## 前言

* 文档介绍了目前昆仑支持的模型以及如何在昆仑设备上训练这些模型。支持昆仑的pddlePaddle安装参考install_kunlun(https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/install/install_Kunlun_zh.md)

## 昆仑训练
* 数据来源参考[ImageNet1k](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/docs/en/tutorials/data_en.md)。昆仑训练效果与CPU/GPU对齐。

### ResNet50
* 命令：

```python3.7 tools/train_multi_platform.py -c configs/kunlun/ResNet50.yaml -o use_gpu=False -o use_xpu=True```

与cpu/gpu训练的区别是加上-o use_xpu=True, 表示执行在昆仑设备上。
