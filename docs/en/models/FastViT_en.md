# FastViT series
---

## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>

## 1. Overview

FastViT is a family of fast inference hybrid neural networks for image classification tasks. It combines the efficiency of convolutional neural networks with the powerful representation learning capabilities of vision transformers. The design focuses on optimizing inference speed while maintaining high accuracy, making it suitable for real-time applications on edge devices.

FastViT achieves this through several key innovations:
- **MobileOne Block**: A re-parameterizable convolutional block that enables training-time multi-branch architecture while allowing efficient inference-time single-branch execution
- **Token Mixer**: A hybrid token mixing mechanism that combines convolutional token mixing with self-attention for efficient spatial information processing
- **Reparameterization**: Converts multi-branch training architecture to single-branch inference architecture for deployment efficiency

Reference: [FastViT: A Fast Hybrid Vision Transformer for Mobile Devices](https://arxiv.org/abs/2303.14189)

<a name='2'></a>

## 2. Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(M) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FastViT_T8   | 0.7598 | 0.9269 | 0.766 | 0.929 | 305  | 7.8 |
| FastViT_T12  | 0.7810 | 0.9371 | 0.786 | 0.940 | 406  | 9.2 |
| FastViT_SA12 | 0.7934 | 0.9446 | 0.800 | 0.947 | 658  | 11 |
| FastViT_SA24 | 0.8085 | 0.9551 | 0.816 | 0.960 | 1120 | 19 |
| FastViT_SA36 | 0.8191 | 0.9551 | 0.826 | 0.960 | 2353 | 39 |
| FastViT_MA36 | 0.8247 | 0.9551 | 0.831 | 0.960 | 2353 | 39 |

**Note**: The accuracy values are from the original paper and may vary slightly depending on preprocessing and evaluation settings.
