# RepVGG series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview

RepVGG (Making VGG-style ConvNets Great Again) series model is a simple but powerful convolutional neural network architecture proposed by Tsinghua University (Guiguang Ding's team), MEGVII Technology (Jian Sun et al.), HKUST and Aberystwyth University in 2021. The architecture has an inference time agent similar to VGG. The main body is composed of 3x3 convolution and relu stack, while the training time model has multi branch topology. The decoupling of training time and inference time is realized by re-parameterization technology, so the model is called repvgg. [paper](https://arxiv.org/abs/2101.03697).

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models | Top1 | Top5 | Reference<br>top1| FLOPs<br>(G) |
|:--:|:--:|:--:|:--:|:--:|
| RepVGG_A0 | 0.7131 | 0.9016 | 0.7241 |     |
| RepVGG_A1 | 0.7380 | 0.9146 | 0.7446 |     |
| RepVGG_A2 | 0.7571 | 0.9264 | 0.7648 |     |
| RepVGG_B0 | 0.7450 | 0.9213 | 0.7514 |     |
| RepVGG_B1 | 0.7773 | 0.9385 | 0.7837 |     |
| RepVGG_B2 | 0.7813 | 0.9410 | 0.7878 |     |
| RepVGG_B1g2 | 0.7732 | 0.9359 | 0.7778 |    |
| RepVGG_B1g4 | 0.7675 | 0.9335 | 0.7758 |    |
| RepVGG_B2g4 | 0.7881 | 0.9448 | 0.7938 |    |
| RepVGG_B3g4 | 0.7965 | 0.9485 | 0.8021 |    |

Params, FLOPs, Inference speed and other information are coming soon.
