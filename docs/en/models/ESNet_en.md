# ESNet Series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview

ESNet (Enhanced ShuffleNet) is a lightweight network developed by Baidu. This network combines the advantages of MobileNetV3, GhostNet, and PPLCNet on the basis of ShuffleNetV2 to form a faster and more accurate network on ARM devices, Because of its excellent performance, [PP-PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet) launched in PaddleDetection uses this model as a backbone, with stronger object detection algorithm, the final mAP index refreshed the SOTA index of the object detection model on the ARM device in one fell swoop.

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models | Top1 | Top5 | FLOPs<br>(M) | Params<br/>(M) |
|:--:|:--:|:--:|:--:|:--:|
| ESNet_x0_25 | 62.48 | 83.46 | 30.9 | 2.83 |
| ESNet_x0_5 | 68.82 | 88.04 | 67.3 | 3.25 |
| ESNet_x0_75 | 72.24 | 90.45 | 123.7 | 3.87 |
| ESNet_x1_0 | 73.92 | 91.40 | 197.3 | 4.64 |

Please stay tuned for information such as Inference speed.
