# DLA series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## Overview

DLA (Deep Layer Aggregation). Visual recognition requires rich representations that span levels from low to high, scales from small to large, and resolutions from fine to coarse. Even with the depth of features in a convolutional network, a layer in isolation is not enough: compounding and aggregating these representations improves inference of what and where. Although skip connections have been incorporated to combine layers, these connections have been "shallow" themselves, and only fuse by simple, one-step operations. The authors augment standard architectures with deeper aggregation to better fuse information across layers. Deep layer aggregation structures iteratively and hierarchically merge the feature hierarchy to make networks with better accuracy and fewer parameters. Experiments across architectures and tasks show that deep layer aggregation improves recognition and resolution compared to existing branching and merging schemes.  [paper](https://arxiv.org/abs/1707.06484)

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

|         Model         | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |
|:-----------------:|:----------:|:---------:|:---------:|:---------:|
| DLA34                 | 15.8       | 3.1       | 76.03     |   92.98   |
| DLA46_c              | 1.3        | 0.5       | 63.21     |   85.30   |
| DLA46x_c            | 1.1        | 0.5       | 64.36     |   86.01   |
| DLA60               | 22.0       | 4.2       | 76.10    |   92.92   |
| DLA60x             | 17.4       | 3.5       | 77.53    |   93.78   |
| DLA60x_c              | 1.3        | 0.6       | 66.45     |   87.54   | 
| DLA102                | 33.3       | 7.2       | 78.93     |   94.52   |
| DLA102x             | 26.4       | 5.9       | 78.10     |   94.00   |
| DLA102x2              | 41.4       | 9.3       | 78.85     |   94.45   |
| DLA169                | 53.5       | 11.6      | 78.09    |   94.09   |
