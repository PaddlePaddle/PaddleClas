# CSWinTransformer
---
## 目录

* [1. 概述](#1)
* [2. 精度、FLOPs 和参数量](#2)

<a name='1'></a>

## 1. 概述
CSWinTransformer 是一种新的视觉 Transformer 网络，可以用作计算机视觉领域的通用骨干网路。 CSWinTransformer 提出了通过十字形的窗口来做 self-attention，它不仅计算效率非常高，而且能够通过两层计算就获得全局的感受野。CSWinTransformer 还提出了新的编码方式：LePE，进一步提高了模型的准确率。[论文地址](https://arxiv.org/abs/2107.00652)。

<a name='2'></a>

## 2. 精度、FLOPs 和参数量

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| CSWinTransformer_tiny_224    | 0.8281 | 0.9628 | 0.828 | - | 4.1  | 22   |
| CSWinTransformer_small_224   | 0.8358 | 0.9658 | 0.836 | - | 6.4  | 35   |
| CSWinTransformer_base_224    | 0.8420 | 0.9692 | 0.842 | - | 14.3 | 77   |
| CSWinTransformer_large_224   | 0.8643 | 0.9799 | 0.865 | - | 32.2 | 173.3   |
| CSWinTransformer_base_384   | 0.8550 | 0.9749 | 0.855 | - | 42.2 | 77   |
| CSWinTransformer_large_384   | 0.8748 | 0.9833 | 0.875 | - | 94.7 | 173.3   |
