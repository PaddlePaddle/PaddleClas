# HarDNet系列

## 概述

HarDNet（Harmonic DenseNet）是 2019 年由国立清华大学提出的一种全新的神经网络，在低 MAC 和内存流量的条件下实现了高效率。与 FC-DenseNet-103，DenseNet-264，ResNet-50，ResNet-152 和SSD-VGG 相比，新网络的推理时间减少了 35%，36%，30%，32% 和 45%。我们使用了包括Nvidia Profiler 和 ARM Scale-Sim 在内的工具来测量内存流量，并验证推理延迟确实与内存流量消耗成正比，并且所提议的网络消耗的内存流量很低。[论文地址](https://arxiv.org/abs/1909.00948)。

## 精度、FLOPS和参数量

|         Model        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |
|:---------------------:|:----------:|:---------:|:---------:|:---------:|
| HarDNet68        | 17.6       | 4.3       | 75.46     | 92.65    |
| HarDNet85          | 36.7       | 9.1       | 77.44     |  93.55    |
| HarDNet39_ds       |  3.5       | 0.4       | 71.33     |  89.98    |
| HarDNet68_ds       |  4.2       | 0.8       | 73.62     |  91.52    |
