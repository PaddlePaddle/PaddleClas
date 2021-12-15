# SwinTransformer
---
## 目录

* [1. 概述](#1)
* [2. 精度、FLOPS 和参数量](#2)
* [3. 基于V100 GPU 的预测速度](#3)

<a name='1'></a>

## 1. 概述
Swin Transformer 是一种新的视觉 Transformer 网络，可以用作计算机视觉领域的通用骨干网路。SwinTransformer 由移动窗口（shifted windows）表示的层次 Transformer 结构组成。移动窗口将自注意计算限制在非重叠的局部窗口上，同时允许跨窗口连接，从而提高了网络性能。[论文地址](https://arxiv.org/abs/2103.14030)。

<a name='2'></a>

## 2. 精度、FLOPS 和参数量

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SwinTransformer_tiny_patch4_window7_224    | 0.8069 | 0.9534 | 0.812 | 0.955 | 4.5  | 28   |
| SwinTransformer_small_patch4_window7_224   | 0.8275 | 0.9613 | 0.832 | 0.962 | 8.7  | 50   |
| SwinTransformer_base_patch4_window7_224    | 0.8300 | 0.9626 | 0.835 | 0.965 | 15.4 | 88   |
| SwinTransformer_base_patch4_window12_384   | 0.8439 | 0.9693 | 0.845 | 0.970 | 47.1 | 88   |
| SwinTransformer_base_patch4_window7_224<sup>[1]</sup>    | 0.8487 | 0.9746 | 0.852 | 0.975 | 15.4 | 88   |
| SwinTransformer_base_patch4_window12_384<sup>[1]</sup>   | 0.8642 | 0.9807 | 0.864 | 0.980 | 47.1 | 88   |
| SwinTransformer_large_patch4_window7_224<sup>[1]</sup>   | 0.8596 | 0.9783 | 0.863 | 0.979 | 34.5 | 197 |
| SwinTransformer_large_patch4_window12_384<sup>[1]</sup>  | 0.8719 | 0.9823 | 0.873 | 0.982 | 103.9 | 197 |

[1]：基于 ImageNet22k 数据集预训练，然后在 ImageNet1k 数据集迁移学习得到。

**注**：与 Reference 的精度差异源于数据预处理不同。

<a name='3'></a>

## 3. 基于 V100 GPU 的预测速度

| Models                                                  | Crop Size | Resize Short Size | FP32<br/>Batch Size=1<br/>(ms) | FP32<br/>Batch Size=4<br/>(ms) | FP32<br/>Batch Size=8<br/>(ms) |
| ------------------------------------------------------- | --------- | ----------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| SwinTransformer_tiny_patch4_window7_224                 | 224       | 256               | 6.59                           | 9.68                           | 16.32                          |
| SwinTransformer_small_patch4_window7_224                | 224       | 256               | 12.54                          | 17.07                          | 28.08                          |
| SwinTransformer_base_patch4_window7_224                 | 224       | 256               | 13.37                          | 23.53                          | 39.11                          |
| SwinTransformer_base_patch4_window12_384                | 384       | 384               | 19.52                          | 64.56                          | 123.30                         |
| SwinTransformer_base_patch4_window7_224<sup>[1]</sup>   | 224       | 256               | 13.53                          | 23.46                          | 39.13                          |
| SwinTransformer_base_patch4_window12_384<sup>[1]</sup>  | 384       | 384               | 19.65                          | 64.72                          | 123.42                         |
| SwinTransformer_large_patch4_window7_224<sup>[1]</sup>  | 224       | 256               | 15.74                          | 38.57                          | 71.49                          |
| SwinTransformer_large_patch4_window12_384<sup>[1]</sup> | 384       | 384               | 32.61                          | 116.59                         | 223.23                         |

[1]：基于 ImageNet22k 数据集预训练，然后在 ImageNet1k 数据集迁移学习得到。
