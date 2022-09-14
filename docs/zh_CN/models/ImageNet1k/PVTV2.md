# PVTV2

---

## 目录

* [1. 概述](#1)
* [2. 精度、FLOPS 和参数量](#2)

<a name='1'></a>

## 1. 概述

PVTV2 是 VisionTransformer 系列模型，该模型基于 PVT（Pyramid Vision Transformer）改进得到，PVT 模型使用 Transformer 结构构建了特征金字塔网络。PVTV2 的主要创新点有：1. 带 overlap 的 Patch embeding；2. 结合卷积神经网络；3. 注意力模块为线性复杂度。[论文地址](https://arxiv.org/pdf/2106.13797.pdf)。

<a name='2'></a>
## 2. 精度、FLOPS 和参数量

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| PVT_V2_B0 | 0.7052 | 0.9016 | 0.705 | - | 0.53 | 3.7 |
| PVT_V2_B1 | 0.7869 | 0.9450 | 0.787 | - | 2.0 | 14.0 |
| PVT_V2_B2 | 0.8206 | 0.9599 | 0.820 | - | 3.9 | 25.4 |
| PVT_V2_B3 | 0.8310 | 0.9648 | 0.831 | - | 6.7 | 45.2 |
| PVT_V2_B4 | 0.8361 | 0.9666 | 0.836 | - | 9.8 | 62.6 |
| PVT_V2_B5 | 0.8374 | 0.9662 | 0.838 | - | 11.4 | 82.0 |
| PVT_V2_B2_Linear | 0.8205 | 0.9605 | 0.820 | - | 3.8 | 22.6 |
