# PVTV2

---

## Content

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview

PVTV2 is VisionTransformer series model, which build on PVT (Pyramid Vision Transformer). PVT use Transformer block to build feature pyramid network. The mainly designs of PVTV2 are: (1) overlapping patch embedding, (2) convolutional feedforward networks, and (3) linear complexity attention layers. [Paper](https://arxiv.org/pdf/2106.13797.pdf).

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| PVT_V2_B0 | 0.705 | 0.902 | 0.705 | - | 0.53 | 3.7 |
| PVT_V2_B1 | 0.787 | 0.945 | 0.787 | - | 2.0 | 14.0 |
| PVT_V2_B2 | 0.821 | 0.960 | 0.820 | - | 3.9 | 25.4 |
| PVT_V2_B3 | 0.831 | 0.965 | 0.831 | - | 6.7 | 45.2 |
| PVT_V2_B4 | 0.836 | 0.967 | 0.836 | - | 9.8 | 62.6 |
| PVT_V2_B5 | 0.837 | 0.966 | 0.838 | - | 11.4 | 82.0 |
| PVT_V2_B2_Linear | 0.821 | 0.961 | 0.821 | - | 3.8 | 22.6 |
