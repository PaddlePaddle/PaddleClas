# TNT 系列
---
## 目录

* [1. 概述](#1)
* [2. 精度、FLOPS 和参数量](#2)

<a name='1'></a>

## 1. 概述

TNT(Transformer-iN-Transformer)系列模型由华为诺亚于 2021 年提出，用于对 patch 级别和 pixel 级别的表示进行建模。在每个 TNT 块中，outer transformer block 用于处理 patch 嵌入，inner transformer block 从 pixel 嵌入中提取局部特征。通过线性变换层将 pixel 级特征投影到 patch 嵌入空间，然后加入到 patch 中。通过对 TNT 块的叠加，建立了用于图像识别的 TNT 模型。在 ImageNet 基准测试和下游任务上的实验证明了该 TNT 体系结构的优越性和有效性。例如，在计算量相当的情况下 TNT 能在 ImageNet 上达到 81.3% 的 top-1 精度，比 DeiT 高 1.5%。[论文地址](https://arxiv.org/abs/2103.00112)。

<a name='2'></a>

## 2. 精度、FLOPS 和参数量

|         Model        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |
|:---------------------:|:----------:|:---------:|:---------:|:---------:|
|        TNT_small        | 23.8       | 5.2       | 81.21     |   95.63   |
