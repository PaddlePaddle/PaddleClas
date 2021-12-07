# HarDNet series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview

HarDNet（Harmonic DenseNet）is a brand new neural network proposed by National Tsing Hua University in 2019, which to achieve high efficiency in terms of both low MACs and memory traffic. The new network achieves 35%, 36%, 30%, 32%, and 45% inference time reduction compared with FC-DenseNet-103, DenseNet-264, ResNet-50, ResNet-152, and SSD-VGG, respectively. We use tools including Nvidia profiler and ARM Scale-Sim to measure the memory traffic and verify that the inference latency is indeed proportional to the memory traffic consumption and the proposed network consumes low memory traffic. [Paper](https://arxiv.org/abs/1909.00948).

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

|         Model        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |
|:---------------------:|:----------:|:---------:|:---------:|:---------:|
| HarDNet68        | 17.6       | 4.3       | 75.46     | 92.65    |
| HarDNet85          | 36.7       | 9.1       | 77.44     |  93.55    |
| HarDNet39_ds       |  3.5       | 0.4       | 71.33     |  89.98    |
| HarDNet68_ds       |  4.2       | 0.8       | 73.62     |  91.52    |