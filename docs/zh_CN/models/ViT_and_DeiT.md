# ViT 与 DeiT 系列
---
## 目录

* [1. 概述](#1)
* [2. 精度、FLOPS 和参数量](#2)

<a name='1'></a>

## 1. 概述

ViT（Vision Transformer）系列模型是 Google 在 2020 年提出的，该模型仅使用标准的 Transformer 结构，完全抛弃了卷积结构，将图像拆分为多个 patch 后再输入到 Transformer 中，展示了 Transformer 在 CV 领域的潜力。[论文地址](https://arxiv.org/abs/2010.11929)。

DeiT（Data-efficient Image Transformers）系列模型是由 FaceBook 在 2020 年底提出的，针对 ViT 模型需要大规模数据集训练的问题进行了改进，最终在 ImageNet 上取得了 83.1%的 Top1 精度。并且使用卷积模型作为教师模型，针对该模型进行知识蒸馏，在 ImageNet 数据集上可以达到 85.2% 的 Top1 精度。[论文地址](https://arxiv.org/abs/2012.12877)。



<a name='2'></a>
## 2. 精度、FLOPS 和参数量

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ViT_small_patch16_224 | 0.7769 | 0.9342 | 0.7785 | 0.9342 |      |      |
| ViT_base_patch16_224  | 0.8195 | 0.9617 | 0.8178 | 0.9613 |      |      |
| ViT_base_patch16_384  | 0.8414 | 0.9717 | 0.8420 | 0.9722 |      |      |
| ViT_base_patch32_384  | 0.8176 | 0.9613 | 0.8166 | 0.9613 |      |      |
| ViT_large_patch16_224 | 0.8323 | 0.9650 | 0.8306 | 0.9644 |      |      |
| ViT_large_patch16_384 | 0.8513 | 0.9736 | 0.8517 | 0.9736 |      |      |
| ViT_large_patch32_384 | 0.8153 | 0.9608 | 0.815  | -      |      |      |


| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeiT_tiny_patch16_224            | 0.718 | 0.910 | 0.722 | 0.911 |      |      |
| DeiT_small_patch16_224           | 0.796 | 0.949 | 0.799 | 0.950 |      |      |
| DeiT_base_patch16_224            | 0.817 | 0.957 | 0.818 | 0.956 |      |      |
| DeiT_base_patch16_384            | 0.830 | 0.962 | 0.829 | 0.972 |      |      |
| DeiT_tiny_distilled_patch16_224  | 0.741 | 0.918 | 0.745 | 0.919 |      |      |
| DeiT_small_distilled_patch16_224 | 0.809 | 0.953 | 0.812 | 0.954 |      |      |
| DeiT_base_distilled_patch16_224  | 0.831 | 0.964 | 0.834 | 0.965 |      |      |
| DeiT_base_distilled_patch16_384  | 0.851 | 0.973 | 0.852 | 0.972 |      |      |

关于 Params、FLOPs、Inference speed 等信息，敬请期待。
