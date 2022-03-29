# CSWinTransformer
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>

## 1. Overview
CSWinTransformer is a new visual Transformer network that can be used as a general backbone network in the field of computer vision. CSWinTransformer proposes to do self-attention through a cross-shaped window, which not only has a very high computational efficiency, but also can obtain a global receptive field through two-layer calculation. CSWinTransformer also proposed a new encoding method: LePE, which further improved the accuracy of the model. [Paper](https://arxiv.org/abs/2107.00652)

<a name='2'></a>

## 2. Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| CSWinTransformer_tiny_224    | 0.8281 | 0.9628 | 0.828 | - | 4.1  | 22   |
| CSWinTransformer_small_224   | 0.8358 | 0.9658 | 0.836 | - | 6.4  | 35   |
| CSWinTransformer_base_224    | 0.8420 | 0.9692 | 0.842 | - | 14.3 | 77   |
| CSWinTransformer_large_224   | 0.8643 | 0.9799 | 0.865 | - | 32.2 | 173.3   |
| CSWinTransformer_base_384   | 0.8550 | 0.9749 | 0.855 | - | 42.2 | 77   |
| CSWinTransformer_large_384   | 0.8748 | 0.9833 | 0.875 | - | 94.7 | 173.3   |
