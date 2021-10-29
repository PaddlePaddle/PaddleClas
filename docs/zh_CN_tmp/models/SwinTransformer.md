# SwinTransformer

## 概述
Swin Transformer 是一种新的视觉Transformer网络，可以用作计算机视觉领域的通用骨干网路。SwinTransformer由移动窗口（shifted windows）表示的层次Transformer结构组成。移动窗口将自注意计算限制在非重叠的局部窗口上，同时允许跨窗口连接，从而提高了网络性能。[论文地址](https://arxiv.org/abs/2103.14030)。


## 精度、FLOPS和参数量

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

[1]：基于ImageNet22k数据集预训练，然后在ImageNet1k数据集迁移学习得到。

**注**：与Reference的精度差异源于数据预处理不同。
