# MobileviT
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>

## 1. Overview

MobileViT is a lightweight visual Transformer network that can be used as a general backbone network in the field of computer vision. MobileViT combines the advantages of CNN and Transformer, which can better deal with global features and local features, and better solve the problem of lack of inductive bias in Transformer models.
, and finally, under the same amount of parameters, compared with other SOTA models, the tasks of image classification, object detection, and semantic segmentation have been greatly improved. [Paper](https://arxiv.org/pdf/2110.02178.pdf)

<a name='2'></a>

## 2. Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(M) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MobileViT_XXS    | 0.6867 | 0.8878 | 0.690 | - | 337.24  | 1.28   |
| MobileViT_XS    | 0.7454 | 0.9227 | 0.747 | - | 930.75  | 2.33   |
| MobileViT_S    | 0.7814 | 0.9413 | 0.783 | - | 1849.35  | 5.59   |
