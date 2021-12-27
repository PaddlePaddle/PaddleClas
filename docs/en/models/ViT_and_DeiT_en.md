# ViT and DeiT series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview

ViT(Vision Transformer) series models were proposed by Google in 2020. These models only use the standard transformer structure, completely abandon the convolution structure, splits the image into multiple patches and then inputs them into the transformer, showing the potential of transformer in the CV field.。[Paper](https://arxiv.org/abs/2010.11929)。

DeiT(Data-efficient Image Transformers) series models were proposed by Facebook at the end of 2020. Aiming at the problem that the ViT models need large-scale dataset training, the DeiT improved them, and finally achieved 83.1% Top1 accuracy on ImageNet. More importantly, using convolution model as teacher model, and performing knowledge distillation on these models, the Top1 accuracy of 85.2% can be achieved on the ImageNet dataset.

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ViT_small_patch16_224 | 0.7769 | 0.9342 | 0.7785 | 0.9342 |      |      |
| ViT_base_patch16_224  | 0.8195 | 0.9617 | 0.8178 | 0.9613 |      |      |
| ViT_base_patch16_384  | 0.8414 | 0.9717 | 0.8420 | 0.9722 |      |      |
| ViT_base_patch32_384  | 0.8176 | 0.9613 | 0.8166 | 0.9613 |      |      |
| ViT_large_patch16_224 | 0.8323 | 0.9650 | 0.8306 | 0.9644 |      |      |
| ViT_large_patch16_384 | 0.8513 | 0.9736 | 0.8517 | 0.9736 |      |      |
| ViT_large_patch32_384 | 0.8153 | 0.9608 | 0.815  | -      |      |      |


| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeiT_tiny_patch16_224            | 0.718 | 0.910 | 0.722 | 0.911 |      |      |
| DeiT_small_patch16_224           | 0.796 | 0.949 | 0.799 | 0.950 |      |      |
| DeiT_base_patch16_224            | 0.817 | 0.957 | 0.818 | 0.956 |      |      |
| DeiT_base_patch16_384            | 0.830 | 0.962 | 0.829 | 0.972 |      |      |
| DeiT_tiny_distilled_patch16_224  | 0.741 | 0.918 | 0.745 | 0.919 |      |      |
| DeiT_small_distilled_patch16_224 | 0.809 | 0.953 | 0.812 | 0.954 |      |      |
| DeiT_base_distilled_patch16_224  | 0.831 | 0.964 | 0.834 | 0.965 |      |      |
| DeiT_base_distilled_patch16_384  | 0.851 | 0.973 | 0.852 | 0.972 |      |      |


Params, FLOPs, Inference speed and other information are coming soon.
