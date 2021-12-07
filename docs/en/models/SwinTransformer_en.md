# SwinTransformer
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview
Swin Transformer a new vision Transformer, that capably serves as a general-purpose backbone for computer vision. It is a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. [Paper](https://arxiv.org/abs/2103.14030)。

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SwinTransformer_tiny_patch4_window7_224    | 0.8069 | 0.9534 | 0.812 | 0.955 | 4.5  | 28   |
| SwinTransformer_small_patch4_window7_224   | 0.8275 | 0.9613 | 0.832 | 0.962 | 8.7  | 50   |
| SwinTransformer_base_patch4_window7_224    | 0.8300 | 0.9626 | 0.835 | 0.965 | 15.4 | 88   |
| SwinTransformer_base_patch4_window12_384   | 0.8439 | 0.9693 | 0.845 | 0.970 | 47.1 | 88   |
| SwinTransformer_base_patch4_window7_224<sup>[1]</sup>    | 0.8487 | 0.9746 | 0.852 | 0.975 | 15.4 | 88   |
| SwinTransformer_base_patch4_window12_384<sup>[1]</sup>   | 0.8642 | 0.9807 | 0.864 | 0.980 | 47.1 | 88   |
| SwinTransformer_large_patch4_window7_224<sup>[1]</sup>   | 0.8596 | 0.9783 | 0.863 | 0.979 | 34.5 | 197  |
| SwinTransformer_large_patch4_window12_384<sup>[1]</sup>  | 0.8719 | 0.9823 | 0.873 | 0.982 | 103.9 | 197 |

[1]: Based on imagenet22k dataset pre-training, and then in imagenet1k dataset transfer learning.

**Note**: The difference of precision with reference from the difference of data preprocessing.
