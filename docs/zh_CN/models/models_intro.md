# 模型库概览
---
## 目录

* [1. 概述](#1)
* [2. 评估环境](#2)
* [3. 预训练模型列表及下载地址](#3)
* [4. 参考文献](#4)

<a name='1'></a>

## 1. 概述

基于 ImageNet1k 分类数据集，PaddleClas 支持的 36 种系列分类网络结构以及对应的 175 个图像分类预训练模型如下所示，训练技巧、每个系列网络结构的简单介绍和性能评估将在相应章节展现。

 <a name='2'></a>

## 2. 评估环境
* Arm 的评估环境基于骁龙 855（SD855）。
* Intel CPU 的评估环境基于 Intel(R) Xeon(R) Gold 6148。
* GPU 评估环境基于 V100 和 TensorRT。

<div align="center">
    <img src="../../images/models/V100_benchmark/v100.fp32.bs1.main_fps_top1_s.png" width="800">
</div>

<div align="center">
    <img src="../../images/models/mobile_arm_top1.png" width="800">
</div>

<div align="center">
    <img src="../../images/models/V100_benchmark/v100.fp32.bs1.visiontransformer.png" width="800">
</div>

> 如果您觉得此文档对您有帮助，欢迎 star 我们的项目：[https://github.com/PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas)

<a name='3'></a>

## 3. 预训练模型列表及下载地址

- ResNet 及其 Vd 系列
  - ResNet 系列<sup>[[1](#ref1)]</sup>([论文地址](http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html))
    - [ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_pretrained.pdparams)
    - [ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet34_pretrained.pdparams)
    - [ResNet50](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams)
    - [ResNet101](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_pretrained.pdparams)
    - [ResNet152](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet152_pretrained.pdparams)
  - ResNet_vc、ResNet_vd 系列<sup>[[2](#ref2)]</sup>([论文地址](https://arxiv.org/abs/1812.01187))
    - [ResNet50_vc](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vc_pretrained.pdparams)
    - [ResNet18_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_vd_pretrained.pdparams)
    - [ResNet34_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet34_vd_pretrained.pdparams)
    - [ResNet34_vd_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet34_vd_ssld_pretrained.pdparams)
    - [ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams)
    - [ResNet50_vd_v2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_v2_pretrained.pdparams)
    - [ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_vd_pretrained.pdparams)
    - [ResNet152_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet152_vd_pretrained.pdparams)
    - [ResNet200_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet200_vd_pretrained.pdparams)
    - [ResNet50_vd_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams)
    - [Fix_ResNet50_vd_ssld_v2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Fix_ResNet50_vd_ssld_v2_pretrained.pdparams)
    - [ResNet101_vd_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_vd_ssld_pretrained.pdparams)


- 轻量级模型系列
  - PP-LCNet 系列<sup>[[28](#ref28)]</sup>([论文地址](https://arxiv.org/pdf/2109.15099.pdf))
    - [PPLCNet_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams)
     - [PPLCNet_x0_35](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams)
    - [PPLCNet_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams)
    - [PPLCNet_x0_75](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams)
    - [PPLCNet_x1_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams)
    - [PPLCNet_x1_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams)
    - [PPLCNet_x2_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams)
    - [PPLCNet_x2_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams)
    - [PPLCNet_x0_5_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_ssld_pretrained.pdparams)
    - [PPLCNet_x1_0_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_ssld_pretrained.pdparams)
    - [PPLCNet_x2_5_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_ssld_pretrained.pdparams)
  - MobileNetV3 系列<sup>[[3](#ref3)]</sup>([论文地址](https://arxiv.org/abs/1905.02244))
    - [MobileNetV3_large_x0_35](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_35_pretrained.pdparams)
    - [MobileNetV3_large_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams)
    - [MobileNetV3_large_x0_75](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_75_pretrained.pdparams)
    - [MobileNetV3_large_x1_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams)
    - [MobileNetV3_large_x1_25](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_25_pretrained.pdparams)
    - [MobileNetV3_small_x0_35](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_35_pretrained.pdparams)
    - [MobileNetV3_small_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_5_pretrained.pdparams)
    - [MobileNetV3_small_x0_75](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_75_pretrained.pdparams)
    - [MobileNetV3_small_x1_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_0_pretrained.pdparams)
    - [MobileNetV3_small_x1_25](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_25_pretrained.pdparams)
    - [MobileNetV3_large_x1_0_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_ssld_pretrained.pdparams)
    - [MobileNetV3_large_x1_0_ssld_int8]()(coming soon)
    - [MobileNetV3_small_x1_0_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_0_ssld_pretrained.pdparams)
  - MobileNetV2 系列<sup>[[4](#ref4)]</sup>([论文地址](https://arxiv.org/abs/1801.04381))
    - [MobileNetV2_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_25_pretrained.pdparams)
    - [MobileNetV2_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_5_pretrained.pdparams)
    - [MobileNetV2_x0_75](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_75_pretrained.pdparams)
    - [MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams)
    - [MobileNetV2_x1_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x1_5_pretrained.pdparams)
    - [MobileNetV2_x2_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x2_0_pretrained.pdparams)
    - [MobileNetV2_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_ssld_pretrained.pdparams)
  - MobileNetV1 系列<sup>[[5](#ref5)]</sup>([论文地址](https://arxiv.org/abs/1704.04861))
    - [MobileNetV1_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_25_pretrained.pdparams)
    - [MobileNetV1_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_5_pretrained.pdparams)
    - [MobileNetV1_x0_75](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_75_pretrained.pdparams)
    - [MobileNetV1](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_pretrained.pdparams)
    - [MobileNetV1_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_ssld_pretrained.pdparams)
  - ShuffleNetV2 系列<sup>[[6](#ref6)]</sup>([论文地址](https://arxiv.org/abs/1807.11164))
    - [ShuffleNetV2_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_25_pretrained.pdparams)
    - [ShuffleNetV2_x0_33](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_33_pretrained.pdparams)
    - [ShuffleNetV2_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_5_pretrained.pdparams)
    - [ShuffleNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_0_pretrained.pdparams)
    - [ShuffleNetV2_x1_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_5_pretrained.pdparams)
    - [ShuffleNetV2_x2_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x2_0_pretrained.pdparams)
    - [ShuffleNetV2_swish](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_swish_pretrained.pdparams)
  - GhostNet 系列<sup>[[23](#ref23)]</sup>([论文地址](https://arxiv.org/pdf/1911.11907.pdf))
    - [GhostNet_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x0_5_pretrained.pdparams)
    - [GhostNet_x1_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_0_pretrained.pdparams)
    - [GhostNet_x1_3](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_pretrained.pdparams)
    - [GhostNet_x1_3_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_ssld_pretrained.pdparams)
  - MixNet 系列<sup>[[29](#ref29)]</sup>([论文地址](https://arxiv.org/pdf/1907.09595.pdf))
    - [MixNet_S](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_S_pretrained.pdparams)
    - [MixNet_M](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_M_pretrained.pdparams)
    - [MixNet_L](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_L_pretrained.pdparams)
  - ReXNet 系列<sup>[[30](#ref30)]</sup>([论文地址](https://arxiv.org/pdf/2007.00992.pdf))
    - [ReXNet_1_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_0_pretrained.pdparams)
    - [ReXNet_1_3](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_3_pretrained.pdparams)
    - [ReXNet_1_5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_5_pretrained.pdparams)
    - [ReXNet_2_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_2_0_pretrained.pdparams)
    - [ReXNet_3_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_3_0_pretrained.pdparams)


- SEResNeXt 与 Res2Net 系列
  - ResNeXt 系列<sup>[[7](#ref7)]</sup>([论文地址](https://arxiv.org/abs/1611.05431))
    - [ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_32x4d_pretrained.pdparams)
    - [ResNeXt50_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_64x4d_pretrained.pdparams)
    - [ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x4d_pretrained.pdparams)
    - [ResNeXt101_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_64x4d_pretrained.pdparams)
    - [ResNeXt152_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_32x4d_pretrained.pdparams)
    - [ResNeXt152_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_64x4d_pretrained.pdparams)
  - ResNeXt_vd 系列
    - [ResNeXt50_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_vd_32x4d_pretrained.pdparams)
    - [ResNeXt50_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_vd_64x4d_pretrained.pdparams)
    - [ResNeXt101_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_vd_32x4d_pretrained.pdparams)
    - [ResNeXt101_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_vd_64x4d_pretrained.pdparams)
    - [ResNeXt152_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_vd_32x4d_pretrained.pdparams)
    - [ResNeXt152_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_vd_64x4d_pretrained.pdparams)
  - SE_ResNet_vd 系列<sup>[[8](#ref8)]</sup>([论文地址](https://arxiv.org/abs/1709.01507))
    - [SE_ResNet18_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNet18_vd_pretrained.pdparams)
    - [SE_ResNet34_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNet34_vd_pretrained.pdparams)
    - [SE_ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNet50_vd_pretrained.pdparams)
  - SE_ResNeXt 系列
    - [SE_ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt50_32x4d_pretrained.pdparams)
    - [SE_ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt101_32x4d_pretrained.pdparams)
  - SE_ResNeXt_vd 系列
    - [SE_ResNeXt50_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt50_vd_32x4d_pretrained.pdparams)
    - [SENet154_vd](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SENet154_vd_pretrained.pdparams)
  - Res2Net 系列<sup>[[9](#ref9)]</sup>([论文地址](https://arxiv.org/abs/1904.01169))
    - [Res2Net50_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_26w_4s_pretrained.pdparams)
    - [Res2Net50_vd_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_vd_26w_4s_pretrained.pdparams)
    - [Res2Net50_vd_26w_4s_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_vd_26w_4s_ssld_pretrained.pdparams)
    - [Res2Net50_14w_8s](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_14w_8s_pretrained.pdparams)
    - [Res2Net101_vd_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net101_vd_26w_4s_pretrained.pdparams)
    - [Res2Net101_vd_26w_4s_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net101_vd_26w_4s_ssld_pretrained.pdparams)
    - [Res2Net200_vd_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net200_vd_26w_4s_pretrained.pdparams)
    - [Res2Net200_vd_26w_4s_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net200_vd_26w_4s_ssld_pretrained.pdparams)


- Inception 系列
  - GoogLeNet 系列<sup>[[10](#ref10)]</sup>([论文地址](https://arxiv.org/pdf/1409.4842.pdf))
    - [GoogLeNet](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GoogLeNet_pretrained.pdparams)
  - InceptionV3 系列<sup>[[26](#ref26)]</sup>([论文地址](https://arxiv.org/abs/1512.00567))
    - [InceptionV3](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/InceptionV3_pretrained.pdparams)
  - InceptionV4 系列<sup>[[11](#ref11)]</sup>([论文地址](https://arxiv.org/abs/1602.07261))
    - [InceptionV4](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/InceptionV4_pretrained.pdparams)
  - Xception 系列<sup>[[12](#ref12)]</sup>([论文地址](http://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html))
    - [Xception41](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_pretrained.pdparams)
    - [Xception41_deeplab](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_deeplab_pretrained.pdparams)
    - [Xception65](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_pretrained.pdparams)
    - [Xception65_deeplab](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_deeplab_pretrained.pdparams)
    - [Xception71](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception71_pretrained.pdparams)


- HRNet 系列
  - HRNet 系列<sup>[[13](#ref13)]</sup>([论文地址](https://arxiv.org/abs/1908.07919))
    - [HRNet_W18_C](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W18_C_pretrained.pdparams)
    - [HRNet_W18_C_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W18_C_ssld_pretrained.pdparams)
    - [HRNet_W30_C](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W30_C_pretrained.pdparams)
    - [HRNet_W32_C](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W32_C_pretrained.pdparams)
    - [HRNet_W40_C](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W40_C_pretrained.pdparams)
    - [HRNet_W44_C](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W44_C_pretrained.pdparams)
    - [HRNet_W48_C](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W48_C_pretrained.pdparams)
    - [HRNet_W48_C_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W48_C_ssld_pretrained.pdparams)
    - [HRNet_W64_C](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W64_C_pretrained.pdparams)
    - [SE_HRNet_W64_C_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_HRNet_W64_C_ssld_pretrained.pdparams)

- DPN 与 DenseNet 系列
  - DPN 系列<sup>[[14](#ref14)]</sup>([论文地址](https://arxiv.org/abs/1707.01629))
    - [DPN68](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN68_pretrained.pdparams)
    - [DPN92](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN92_pretrained.pdparams)
    - [DPN98](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN98_pretrained.pdparams)
    - [DPN107](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN107_pretrained.pdparams)
    - [DPN131](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN131_pretrained.pdparams)
  - DenseNet 系列<sup>[[15](#ref15)]</sup>([论文地址](https://arxiv.org/abs/1608.06993))
    - [DenseNet121](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams)
    - [DenseNet161](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams)
    - [DenseNet169](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams)
    - [DenseNet201](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet201_pretrained.pdparams)
    - [DenseNet264](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet264_pretrained.pdparams)


- EfficientNet 与 ResNeXt101_wsl 系列
  - EfficientNet 系列<sup>[[16](#ref16)]</sup>([论文地址](https://arxiv.org/abs/1905.11946))
    - [EfficientNetB0_small](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB0_small_pretrained.pdparams)
    - [EfficientNetB0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB0_pretrained.pdparams)
    - [EfficientNetB1](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB1_pretrained.pdparams)
    - [EfficientNetB2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB2_pretrained.pdparams)
    - [EfficientNetB3](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB3_pretrained.pdparams)
    - [EfficientNetB4](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB4_pretrained.pdparams)
    - [EfficientNetB5](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB5_pretrained.pdparams)
    - [EfficientNetB6](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB6_pretrained.pdparams)
    - [EfficientNetB7](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB7_pretrained.pdparams)
  - ResNeXt101_wsl 系列<sup>[[17](#ref17)]</sup>([论文地址](https://arxiv.org/abs/1805.00932))
    - [ResNeXt101_32x8d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x8d_wsl_pretrained.pdparams)
    - [ResNeXt101_32x16d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x16d_wsl_pretrained.pdparams)
    - [ResNeXt101_32x32d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x32d_wsl_pretrained.pdparams)
    - [ResNeXt101_32x48d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x48d_wsl_pretrained.pdparams)
    - [Fix_ResNeXt101_32x48d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Fix_ResNeXt101_32x48d_wsl_pretrained.pdparams)

- ResNeSt 与 RegNet 系列
  - ResNeSt 系列<sup>[[24](#ref24)]</sup>([论文地址](https://arxiv.org/abs/2004.08955))
    - [ResNeSt50_fast_1s1x64d](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_fast_1s1x64d_pretrained.pdparams)
    - [ResNeSt50](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_pretrained.pdparams)
  - RegNet 系列<sup>[[25](#ref25)]</sup>([paper link](https://arxiv.org/abs/2003.13678))
    - [RegNetX_4GF](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_4GF_pretrained.pdparams)

- Transformer 系列
  - Swin-transformer 系列<sup>[[27](#ref27)]</sup>([论文地址](https://arxiv.org/pdf/2103.14030.pdf))
    - [SwinTransformer_tiny_patch4_window7_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams)
    - [SwinTransformer_small_patch4_window7_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_small_patch4_window7_224_pretrained.pdparams)
    - [SwinTransformer_base_patch4_window7_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window7_224_pretrained.pdparams)
    - [SwinTransformer_base_patch4_window12_384](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window12_384_pretrained.pdparams)
    - [SwinTransformer_base_patch4_window7_224_22k](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window7_224_22k_pretrained.pdparams)
    - [SwinTransformer_base_patch4_window7_224_22kto1k](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window7_224_22kto1k_pretrained.pdparams)
    - [SwinTransformer_large_patch4_window12_384_22k](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_large_patch4_window12_384_22k_pretrained.pdparams)
    - [SwinTransformer_large_patch4_window12_384_22kto1k](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_large_patch4_window12_384_22kto1k_pretrained.pdparams)
    - [SwinTransformer_large_patch4_window7_224_22k](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_large_patch4_window7_224_22k_pretrained.pdparams)
    - [SwinTransformer_large_patch4_window7_224_22kto1k](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_large_patch4_window7_224_22kto1k_pretrained.pdparams)
  - ViT 系列<sup>[[31](#ref31)]</sup>([论文地址](https://arxiv.org/pdf/2010.11929.pdf))
      - [ViT_small_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams)
      - [ViT_base_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams)
      - [ViT_base_patch16_384](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams)
      - [ViT_base_patch32_384](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch32_384_pretrained.pdparams)
      - [ViT_large_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams)
      - [ViT_large_patch16_384](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_384_pretrained.pdparams)
      - [ViT_large_patch32_384](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch32_384_pretrained.pdparams)
  - DeiT 系列<sup>[[32](#ref32)]</sup>([论文地址](https://arxiv.org/pdf/2012.12877.pdf))
      - [DeiT_tiny_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_patch16_224_pretrained.pdparams)
      - [DeiT_small_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_patch16_224_pretrained.pdparams)
      - [DeiT_base_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_224_pretrained.pdparams)
      - [DeiT_base_patch16_384](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_384_pretrained.pdparams)
      - [DeiT_tiny_distilled_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_distilled_patch16_224_pretrained.pdparams)
      - [DeiT_small_distilled_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_distilled_patch16_224_pretrained.pdparams)
      - [DeiT_base_distilled_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_224_pretrained.pdparams)
      - [DeiT_base_distilled_patch16_384](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_384_pretrained.pdparams)
  - LeViT 系列<sup>[[33](#ref33)]</sup>([论文地址](https://arxiv.org/pdf/2104.01136.pdf))
      - [LeViT_128S](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128S_pretrained.pdparams)
      - [LeViT_128](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128_pretrained.pdparams)
      - [LeViT_192](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_192_pretrained.pdparams)
      - [LeViT_256](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_256_pretrained.pdparams)
      - [LeViT_384](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_384_pretrained.pdparams)
  - Twins 系列<sup>[[34](#ref34)]</sup>([论文地址](https://arxiv.org/pdf/2104.13840.pdf))
      - [pcpvt_small](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_small_pretrained.pdparams)
      - [pcpvt_base](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_base_pretrained.pdparams)
      - [pcpvt_large](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_large_pretrained.pdparams)
      - [alt_gvt_small](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_small_pretrained.pdparams)
      - [alt_gvt_base](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_base_pretrained.pdparams)
      - [alt_gvt_large](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_large_pretrained.pdparams)
  - TNT 系列<sup>[[35](#ref35)]</sup>([论文地址](https://arxiv.org/pdf/2103.00112.pdf))
      - [TNT_small](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TNT_small_pretrained.pdparams)

- 其他模型
  - AlexNet 系列<sup>[[18](#ref18)]</sup>([论文地址](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf))
    - [AlexNet](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/AlexNet_pretrained.pdparams)
  - SqueezeNet 系列<sup>[[19](#ref19)]</sup>([论文地址](https://arxiv.org/abs/1602.07360))
    - [SqueezeNet1_0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_0_pretrained.pdparams)
    - [SqueezeNet1_1](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_1_pretrained.pdparams)
  - VGG 系列<sup>[[20](#ref20)]</sup>([论文地址](https://arxiv.org/abs/1409.1556))
    - [VGG11](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG11_pretrained.pdparams)
    - [VGG13](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG13_pretrained.pdparams)
    - [VGG16](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG16_pretrained.pdparams)
    - [VGG19](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG19_pretrained.pdparams)
  - DarkNet 系列<sup>[[21](#ref21)]</sup>([论文地址](https://arxiv.org/abs/1506.02640))
    - [DarkNet53](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DarkNet53_pretrained.pdparams)
  - RepVGG 系列<sup>[[36](#ref36)]</sup>([论文地址](https://arxiv.org/pdf/2101.03697.pdf))
      - [RepVGG_A0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_A0_pretrained.pdparams)
      - [RepVGG_A1](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_A1_pretrained.pdparams)
      - [RepVGG_A2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_A2_pretrained.pdparams)
      - [RepVGG_B0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B0_pretrained.pdparams)
      - [RepVGG_B1s](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B1_pretrained.pdparams)
      - [RepVGG_B2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B2_pretrained.pdparams)
      - [RepVGG_B1g2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B1g2_pretrained.pdparams)
      - [RepVGG_B1g4](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B1g4_pretrained.pdparams)
      - [RepVGG_B2g4](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B2g4_pretrained.pdparams)
      - [RepVGG_B3g4](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B3g4_pretrained.pdparams)
   - HarDNet 系列<sup>[[37](#ref37)]</sup>([论文地址](https://arxiv.org/pdf/1909.00948.pdf))
      - [HarDNet39_ds](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet39_ds_pretrained.pdparams)
      - [HarDNet68_ds](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet68_ds_pretrained.pdparams)
      - [HarDNet68](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet68_pretrained.pdparams)
      - [HarDNet85](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet85_pretrained.pdparams)
   - DLA 系列<sup>[[38](#ref38)]</sup>([论文地址](https://arxiv.org/pdf/1707.06484.pdf))
      - [DLA102](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102_pretrained.pdparams)
      - [DLA102x2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102x2_pretrained.pdparams)
      - [DLA102x](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102x_pretrained.pdparams)
      - [DLA169](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA169_pretrained.pdparams)
      - [DLA34](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA34_pretrained.pdparams)
      - [DLA46_c](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA46_c_pretrained.pdparams)
      - [DLA60](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60_pretrained.pdparams)
      - [DLA60x_c](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60x_c_pretrained.pdparams)
      - [DLA60x](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60x_pretrained.pdparams)
   - RedNet 系列<sup>[[39](#ref39)]</sup>([论文地址](https://arxiv.org/pdf/2103.06255.pdf))
      - [RedNet26](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet26_pretrained.pdparams)
      - [RedNet38](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet38_pretrained.pdparams)
      - [RedNet50](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet50_pretrained.pdparams)
      - [RedNet101](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet101_pretrained.pdparams)
      - [RedNet152](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet152_pretrained.pdparams)



**注意**：以上模型中 EfficientNetB1-B7 的预训练模型转自[pytorch 版 EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)，ResNeXt101_wsl 系列预训练模型转自[官方 repo](https://github.com/facebookresearch/WSL-Images)，剩余预训练模型均基于飞桨训练得到的，并在 configs 里给出了相应的训练超参数。

<a name='4'></a>

##  4. 参考文献


<a name="ref1">[1]</a> He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

<a name="ref2">[2]</a> He T, Zhang Z, Zhang H, et al. Bag of tricks for image classification with convolutional neural networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 558-567.

<a name="ref3">[3]</a> Howard A, Sandler M, Chu G, et al. Searching for mobilenetv3[C]//Proceedings of the IEEE International Conference on Computer Vision. 2019: 1314-1324.

<a name="ref4">[4]</a> Sandler M, Howard A, Zhu M, et al. Mobilenetv2: Inverted residuals and linear bottlenecks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 4510-4520.

<a name="ref5">[5]</a> Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications[J]. arXiv preprint arXiv:1704.04861, 2017.

<a name="ref6">[6]</a> Ma N, Zhang X, Zheng H T, et al. Shufflenet v2: Practical guidelines for efficient cnn architecture design[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 116-131.

<a name="ref7">[7]</a> Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1492-1500.


<a name="ref8">[8]</a> Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141.


<a name="ref9">[9]</a> Gao S, Cheng M M, Zhao K, et al. Res2net: A new multi-scale backbone architecture[J]. IEEE transactions on pattern analysis and machine intelligence, 2019.

<a name="ref10">[10]</a> Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 1-9.


<a name="ref11">[11]</a> Szegedy C, Ioffe S, Vanhoucke V, et al. Inception-v4, inception-resnet and the impact of residual connections on learning[C]//Thirty-first AAAI conference on artificial intelligence. 2017.

<a name="ref12">[12]</a> Chollet F. Xception: Deep learning with depthwise separable convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1251-1258.

<a name="ref13">[13]</a> Wang J, Sun K, Cheng T, et al. Deep high-resolution representation learning for visual recognition[J]. arXiv preprint arXiv:1908.07919, 2019.

<a name="ref14">[14]</a> Chen Y, Li J, Xiao H, et al. Dual path networks[C]//Advances in neural information processing systems. 2017: 4467-4475.

<a name="ref15">[15]</a> Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 4700-4708.


<a name="ref16">[16]</a> Tan M, Le Q V. Efficientnet: Rethinking model scaling for convolutional neural networks[J]. arXiv preprint arXiv:1905.11946, 2019.

<a name="ref17">[17]</a> Mahajan D, Girshick R, Ramanathan V, et al. Exploring the limits of weakly supervised pretraining[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 181-196.

<a name="ref18">[18]</a> Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]//Advances in neural information processing systems. 2012: 1097-1105.

<a name="ref19">[19]</a> Iandola F N, Han S, Moskewicz M W, et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size[J]. arXiv preprint arXiv:1602.07360, 2016.

<a name="ref20">[20]</a> Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

<a name="ref21">[21]</a> Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

<a name="ref22">[22]</a> Ding X, Guo Y, Ding G, et al. Acnet: Strengthening the kernel skeletons for powerful cnn via asymmetric convolution blocks[C]//Proceedings of the IEEE International Conference on Computer Vision. 2019: 1911-1920.

<a name="ref23">[23]</a> Han K, Wang Y, Tian Q, et al. GhostNet: More features from cheap operations[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 1580-1589.

<a name="ref24">[24]</a> Zhang H, Wu C, Zhang Z, et al. Resnest: Split-attention networks[J]. arXiv preprint arXiv:2004.08955, 2020.

<a name="ref25">[25]</a> Radosavovic I, Kosaraju R P, Girshick R, et al. Designing network design spaces[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 10428-10436.

<a name="ref26">[26]</a> C.Szegedy, V.Vanhoucke, S.Ioffe, J.Shlens, and Z.Wojna. Rethinking the inception architecture for computer vision. arXiv preprint arXiv:1512.00567, 2015.

<a name="ref27">[27]</a> Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin and Baining Guo. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.

<a name="ref28">[28]</a>Cheng Cui, Tingquan Gao, Shengyu Wei, Yuning Du, Ruoyu Guo, Shuilong Dong, Bin Lu, Ying Zhou, Xueying Lv, Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LCNet: A Lightweight CPU Convolutional Neural Network.

<a name="ref29">[29]</a>Mingxing Tan, Quoc V. Le. MixConv: Mixed Depthwise Convolutional Kernels.

<a name="ref30">[30]</a>Dongyoon Han, Sangdoo Yun, Byeongho Heo, YoungJoon Yoo. Rethinking Channel Dimensions for Efficient Model Design.

<a name="ref31">[31]</a>Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE.

<a name="ref32">[32]</a>Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Herve Jegou. Training data-efficient image transformers & distillation through attention.

<a name="ref33">[33]</a>Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Herve Jegou, Matthijs Douze. LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference.

<a name="ref34">[34]</a>Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, Chunhua Shen. Twins: Revisiting the Design of Spatial Attention in Vision Transformers.

<a name="ref35">[35]</a>Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, Yunhe Wang. Transformer in Transformer.

<a name="ref36">[36]</a>Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun. RepVGG: Making VGG-style ConvNets Great Again.

<a name="ref37">[37]</a>Ping Chao, Chao-Yang Kao, Yu-Shan Ruan, Chien-Hsiang Huang, Youn-Long Lin. HarDNet: A Low Memory Traffic Network.

<a name="ref38">[38]</a>Fisher Yu, Dequan Wang, Evan Shelhamer, Trevor Darrell. Deep Layer Aggregation.

<a name="ref39">[39]</a>Duo Lim Jie Hu, Changhu Wang, Xiangtai Li, Qi She, Lei Zhu, Tong Zhang, Qifeng Chen. Involution: Inverting the Inherence of Convolution for Visual Recognition.
