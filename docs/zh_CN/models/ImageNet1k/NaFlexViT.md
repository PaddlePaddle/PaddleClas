# NaFlexViT 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 当前支持的模型](#1.2)
- [2. 模型快速体验](#2)
- [3. 模型训练、评估和预测](#3)

<a name='1'></a>

## 1. 模型介绍

<a name='1.1'></a>

### 1.1 模型简介

NaFlexViT 是 `timm` 中面向灵活输入场景实现的一类 Vision Transformer 结构，核心目标是在保持 ViT 主体结构的同时，支持：

- 基于线性 patch embedding 的 patch 表示
- learned / factorized 位置编码
- 变长宽输入下的位置编码插值
- register tokens 与 global average pooling

当前 PaddleClas 中提供的实现以
[`timm` 的 `naflexvit.py`](https://github.com/huggingface/pytorch-image-models/blob/8d0f79effa3dbc922afbfb431fbadd4648938de7/timm/models/naflexvit.py)
为参考，优先覆盖分类主干与前向对齐所需的最小能力集合。当前提供以下能力：

- 与 `timm` 权重进行前向对齐的验证脚本
- ImageNet1k 分类训练配置补充
- 静态图导出支持

当前阶段不提供全量 ImageNet 训练精度与 Paddle 预训练权重下载链接。

<a name='1.2'></a>

### 1.2 当前支持的模型

| Models | 位置编码 | 用途 | 输出形式 | 训练配置 | 前向对齐 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| `naflexvit_base_patch16_gap` | learned | ImageNet1k 分类 | logits | 支持 | 已验证 |
| `naflexvit_base_patch16_par_gap` | learned + aspect preserving | ImageNet1k 分类 | logits | 支持 | 已验证 |
| `naflexvit_base_patch16_parfac_gap` | factorized + aspect preserving | ImageNet1k 分类 | logits | 支持 | 已验证 |

训练配置位于：

- `ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_gap.yaml`
- `ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_par_gap.yaml`
- `ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_parfac_gap.yaml`

前向对齐脚本位于：

- `tools/verify_naflexvit_alignment.py`

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 后，可以参考 [ResNet50 模型快速体验](./ResNet.md#2-模型快速体验) 进行预测。使用 NaFlexViT 时，可将 `Arch.name` 设置为：

- `naflexvit_base_patch16_gap`
- `naflexvit_base_patch16_par_gap`
- `naflexvit_base_patch16_parfac_gap`

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet 数据准备、模型训练、评估和预测等内容。`ppcls/configs/ImageNet/NaFlexViT/` 中已经提供该系列分类模型的训练配置，启动方式可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

当前推荐优先使用以下配置：

- `naflexvit_base_patch16_gap.yaml`
- `naflexvit_base_patch16_par_gap.yaml`
- `naflexvit_base_patch16_parfac_gap.yaml`
