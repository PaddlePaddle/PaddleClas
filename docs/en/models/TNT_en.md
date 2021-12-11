# TNT series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview

TNT(Transformer-iN-Transformer) series models were proposed by Huawei-Noah in 2021 for modeling both patch-level and pixel-level representation.  In each TNT block, an outer transformer block is utilized to process patch embeddings, and an inner transformer block extracts local features from pixel embeddings. The pixel-level feature is projected to the space of patch embedding by a linear transformation layer and then added into the patch. By stacking the TNT blocks, we build the TNT model for image recognition. Experiments on ImageNet benchmark and downstream tasks demonstrate the superiority and efficiency of the proposed TNT architecture. For example, our TNT achieves 81.3% top-1 accuracy on ImageNet which is 1.5% higher than that of DeiT with similar computational cost. [Paper](https://arxiv.org/abs/2103.00112).


<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

|         Model        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |
|:---------------------:|:----------:|:---------:|:---------:|:---------:|
|        TNT_small        | 23.8       | 5.2       | 81.12     |   95.56   |