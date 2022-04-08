# MixFormer: Mixing Features across Windows and Dimensions
---
## 目录

* [1. 概述](#1)
* [2. 精度、FLOPS 和参数量](#2)

<a name='1'></a>
## 1. 概述

MixFormer是一个高效、通用的骨干网路（Vision Transformer）。在MixFormer中，主要有两个创新的设计：（1）通过平行分支的设计，将局部窗口自注意力（local-window self-attention）与Depthwise卷积进行组合，解决局部窗口自注意力的感受野受限的问题，（2）在平行分支之间提出双向交互模块，使得两个分支可以在channel和spatial两个维度都能实现信息互补，增强整体的建模能力。 [paper](https://arxiv.org/abs/2204.02557).

<a name='2'></a>
## 2. 精度、FLOPS 和参数量

| Models | Top1 | Top5 | Reference<br>top1| FLOPs<br>(G) |
|:--:|:--:|:--:|:--:|:--:|
| MixFormer-B0 | - | - | 0.765 |  0.4  |
| MixFormer-B1 | - | - | 0.789 |  0.7  |
| MixFormer-B2 | - | - | 0.800 |  0.9  |
| MixFormer-B3 | - | - | 0.817 |  1.9  |
| MixFormer-B4 | - | - | 0.830 |  3.6  |
| MixFormer-B5 | - | - | 0.835 |  6.8  |
| MixFormer-B6 | - | - | 0.838 |  12.7  |

模型后续将提供下载。
