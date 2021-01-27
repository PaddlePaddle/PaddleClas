# ViT与DeiT系列

## 概述

ViT（Vision Transformer）系列模型是Google在2020年提出的，该模型仅使用标准的Transformer结构，完全抛弃了卷积结构，将图像拆分为多个patch后再输入到Transformer中，展示了Transformer在CV领域的潜力。[论文地址](https://arxiv.org/abs/2010.11929)。

DeiT（Data-efficient Image Transformers）系列模型是由FaceBook在2020年底提出的，针对ViT模型需要大规模数据集训练的问题进行了改进，最终在ImageNet上取得了83.1%的Top1精度。并且使用卷积模型作为教师模型，针对该模型进行知识蒸馏，在ImageNet数据集上可以达到85.2%的Top1精度。[论文地址](https://arxiv.org/abs/2012.12877)。




## 精度、FLOPS和参数量

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

关于Params、FLOPs、Inference speed等信息，敬请期待。
