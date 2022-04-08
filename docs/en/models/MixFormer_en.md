# MixFormer: Mixing Features across Windows and Dimensions
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview

MixFormer is an efficient and general-purpose hybrid vision transformer. There are two main designs in MixFormer: (1) combining local-window self-attention and depth-wise convolution in a parallel design, (2) proposing bi-directional interactions across branches to provide complementary clues in channel and spatial dimensions. [paper](https://arxiv.org/abs/2204.02557).

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models | Top1 | Top5 | Reference<br>top1| FLOPs<br>(G) |
|:--:|:--:|:--:|:--:|:--:|
| MixFormer-B0 | - | - | 0.765 |  0.4  |
| MixFormer-B1 | - | - | 0.789 |  0.7  |
| MixFormer-B2 | - | - | 0.800 |  0.9  |
| MixFormer-B3 | - | - | 0.817 |  1.9  |
| MixFormer-B4 | - | - | 0.830 |  3.6  |
| MixFormer-B5 | - | - | 0.835 |  6.8  |
| MixFormer-B6 | - | - | 0.838 |  12.7  |

The models are coming soon.
