# ViT and DeiT series

## Overview

ViT(Vision Transformer) series models were proposed by Google in 2020. These models only use the standard transformer structure, completely abandon the convolution structure, splits the image into multiple patches and then inputs them into the transformer, showing the potential of transformer in the CV field.。[Paper](https://arxiv.org/abs/2010.11929)。

DeiT(Data-efficient Image Transformers) series models were proposed by Facebook at the end of 2020. Aiming at the problem that the ViT models need large-scale dataset training, the DeiT improved them, and finally achieved 83.1% Top1 accuracy on ImageNet. More importantly, using convolution model as teacher model, and performing knowledge distillation on these models, the Top1 accuracy of 85.2% can be achieved on the ImageNet dataset.


## Accuracy, FLOPS and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| ViT_small_patch16_224 | 0.77268 | 0.93190 | 0.77854 | 0.93424 |      |
| ViT_base_patch16_224 | 0.81760 | 0.96134 | 0.81784 | 0.96126 |      |
| ViT_base_patch16_384 | 0.83928 | 0.97100 | 0.84202 | 0.97218 |      |
| ViT_base_patch32_384 | 0.81242 | 0.95980 | 0.81656 | 0.96130 |  |
| ViT_large_patch16_224 | 0.83248 | 0.96580 | 0.83060 | 0.96444 |  |
| ViT_large_patch16_384 | 0.85066 | 0.97408 | 0.85166 | 0.97362 |  |
| ViT_large_patch32_384 | 0.81054 | 0.95958 | 0.815 | - |  |
| ViT_huge_patch16_224 |  |  |  |  |  |
| ViT_huge_patch32_384 |  | |  | |  |
| | | | | | |


| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| DeiT_tiny_patch16_224        | 0.709 | 0.906 | 0.722 | 0.911 |      |
| DeiT_small_patch16_224        | 0.794 | 0.948 | 0.799 | 0.950 |      |
| DeiT_base_patch16_224        | 0.816 | 0.955 | 0.818 | 0.956 |      |
| DeiT_base_patch16_384 | 0.831 | 0.962 | 0.829 | 0.972 |  |
| DeiT_tiny_distilled_patch16_224 | 0.736 | 0.915 | 0.745 | 0.919 |  |
| DeiT_small_distilled_patch16_224 | 0.810 | 0.953 | 0.812 | 0.954 |  |
| DeiT_base_distilled_patch16_224 | 0.830 | 0.963 | 0.834 | 0.965 |  |
| DeiT_base_distilled_patch16_384 | 0.855 | 0.974 | 0.852 | 0.972 |  |
|  |  | |  | |  |


Params, FLOPs, Inference speed and other information are coming soon.
