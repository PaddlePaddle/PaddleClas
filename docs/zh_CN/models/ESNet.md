# ESNet 系列
---
## 目录

* [1. 概述](#1)
* [2. 精度、FLOPs 和参数量](#2)

<a name='1'></a>

## 1. 概述

ESNet(Enhanced ShuffleNet)是百度自研的一个轻量级网络，该网络在 ShuffleNetV2 的基础上融合了 MobileNetV3、GhostNet、PPLCNet 的优点，组合成了一个在 ARM 设备上速度更快、精度更高的网络，由于其出色的表现，所以在 PaddleDetection 推出的 [PP-PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet) 使用了该模型做 backbone，配合更强的目标检测算法，最终的指标一举刷新了目标检测模型在 ARM 设备上的 SOTA 指标。

<a name='2'></a>

## 2.精度、FLOPs 和参数量

| Models | Top1 | Top5 | FLOPs<br>(M) | Params<br/>(M) |
|:--:|:--:|:--:|:--:|:--:|
| ESNet_x0_25 | 62.48 | 83.46 | 30.9 | 2.83 |
| ESNet_x0_5 | 68.82 | 88.04 | 67.3 | 3.25 |
| ESNet_x0_75 | 72.24 | 90.45 | 123.7 | 3.87 |
| ESNet_x1_0 | 73.92 | 91.40 | 197.3 | 4.64 |

关于 Inference speed 等信息，敬请期待。
