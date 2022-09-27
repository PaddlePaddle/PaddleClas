# MobileviT
---
## 目录

* [1. 概述](#1)
* [2. 精度、FLOPs 和参数量](#2)

<a name='1'></a>

## 1. 概述

MobileViT 是一个轻量级的视觉 Transformer 网络，可以用作计算机视觉领域的通用骨干网路。 MobileViT 结合了 CNN 和 Transformer 的优势，可以更好的处理全局特征和局部特征，更好地解决 Transformer 模型缺乏归纳偏置的问题，最终，在同样参数量下，与其他 SOTA 模型相比，在图像分类、目标检测、语义分割任务上都有大幅提升。[论文地址](https://arxiv.org/pdf/2110.02178.pdf)。

<a name='2'></a>

## 2. 精度、FLOPs 和参数量

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(M) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MobileViT_XXS    | 0.6867 | 0.8878 | 0.690 | - | 337.24  | 1.28   |
| MobileViT_XS    | 0.7454 | 0.9227 | 0.747 | - | 930.75  | 2.33   |
| MobileViT_S    | 0.7814 | 0.9413 | 0.783 | - | 1849.35  | 5.59   |
