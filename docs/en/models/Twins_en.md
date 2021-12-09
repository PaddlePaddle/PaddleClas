# Twins
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview
The Twins network includes Twins-PCPVT and Twins-SVT, which focuses on the meticulous design of the spatial attention mechanism, resulting in a simple but more effective solution. Since the architecture only involves matrix multiplication, and the current deep learning framework has a high degree of optimization for matrix multiplication, the architecture is very efficient and easy to implement. Moreover, this architecture can achieve excellent performance in a variety of downstream vision tasks such as image classification, target detection, and semantic segmentation. [Paper](https://arxiv.org/abs/2104.13840).

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models        | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| pcpvt_small   | 0.8082 | 0.9552 | 0.812 | - | 3.7 | 24.1   |
| pcpvt_base    | 0.8242 | 0.9619 | 0.827 | - | 6.4 | 43.8   |
| pcpvt_large   | 0.8273 | 0.9650 | 0.831 | - | 9.5 | 60.9   |
| alt_gvt_small | 0.8140 | 0.9546 | 0.817 | - | 2.8  | 24   |
| alt_gvt_base  | 0.8294 | 0.9621 | 0.832 | - | 8.3  | 56   |
| alt_gvt_large | 0.8331 | 0.9642 | 0.837 | - | 14.8 | 99.2   |

**Note**:The difference in accuracy from Reference is due to the difference in data preprocessing.
