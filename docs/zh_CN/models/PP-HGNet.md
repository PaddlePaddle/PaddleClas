# PP-HGNet 系列
---
## 目录

* [1. 概述](#1)
* [2. 精度、FLOPs 和参数量](#2)

<a name='1'></a>

## 1. 概述

PP-HGNet是百度自研的一个在 GPU 端上高性能的网络，该网络在 VOVNet 的基础上融合了 ResNet_vd、PPLCNet 的优点，使用了可学习的下采样层，组合成了一个在 GPU 设备上速度快、精度高的网络，超越其他 GPU 端 SOTA 模型。

<a name='2'></a>

## 2.精度、FLOPs 和参数量

| Models | Top1 | Top5 | FLOPs<br>(G) | Params<br/>(M) |
|:--:|:--:|:--:|:--:|:--:|
| PPHGNet_tiny | 79.83 | 95.04 | 4.54 | 14.75 |
| PPHGNet_tiny_ssld | 81.95 | 96.12 | 4.54 | 14.75 |
| PPHGNet_small | 81.51 | 95.82 | 8.53 | 24.38 |

关于 Inference speed 等信息，敬请期待。
