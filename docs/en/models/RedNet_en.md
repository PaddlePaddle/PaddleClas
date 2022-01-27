# RedNet series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview

In the backbone of ResNet and in all bottleneck positions of backbone, the convolution is replaced by Involution, but all convolutions are reserved for channel mapping and fusion. These carefully redesigned entities combine to form a new efficient backbone network, called Rednet. [paper](https://arxiv.org/abs/2103.06255).

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

|         Model         | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |
|:---------------------:|:----------:|:---------:|:---------:|:---------:|
| RedNet26             |  9.2       | 1.7       | 75.95     | 93.19     |
| RedNet38            | 12.4       | 2.2       | 77.47     | 93.56     |
| RedNet50             | 15.5       | 2.7       | 78.33     | 94.17     |
| RedNet101           | 25.7       | 4.7       | 78.94     | 94.36     |
| RedNet152           | 34.0       | 6.8       | 79.17     | 94.40     |