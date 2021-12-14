# LeViT series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview
LeViT is a fast inference hybrid neural network for image classification tasks. Its design considers the performance of the network model on different hardware platforms, so it can better reflect the real scenarios of common applications. Through a large number of experiments, the author found a better way to combine the convolutional neural network and the Transformer system, and proposed an attention-based method to integrate the position information encoding in the Transformer. [Paper](https://arxiv.org/abs/2104.01136)。

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(M) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| LeViT-128S | 0.7598 | 0.9269 | 0.766 | 0.929 | 305  | 7.8 |
| LeViT-128  | 0.7810 | 0.9371 | 0.786 | 0.940 | 406  | 9.2 |
| LeViT-192  | 0.7934 | 0.9446 | 0.800 | 0.947 | 658  | 11 |
| LeViT-256  | 0.8085 | 0.9497 | 0.816 | 0.954 | 1120 | 19 |
| LeViT-384  | 0.8191 | 0.9551 | 0.826 | 0.960 | 2353 | 39 |


**Note**：The difference in accuracy from Reference is due to the difference in data preprocessing and the absence of distilled head as output.
