# Foundation ViT介绍文档

## 目录

1. [功能介绍](#1-功能介绍)
2. [使用说明](#2-使用说明)
3. [模型介绍](#3-模型介绍)
4. [参考文献](#4-参考文献)

## 1. 功能介绍

为支持视觉大模型的使用，PaddleClas提供了各系列视觉大模型的预训练权重以及特征提取功能，可使用该功能得到在大数据上完成预训练的视觉大模型特征。

## 2. 使用说明

以模型`CLIP_vit_base_patch16_224`为例，使用该模型以及对应的预训练权重进行特征提取的代码如下：

```python
from ppcls.utils import config
from ppcls.arch import build_model
import paddle
pretrained = './paddle_weights/CLIP_vit_base_patch16_224.pdparams' # path to pretrained weight
cfg = {"Arch": {"name": "CLIP_vit_base_patch16_224"}}
model = build_model(cfg, mode="train")
model.set_state_dict(paddle.load(pretrained))
inputs = paddle.randn((1,3,224,224))  # create input
output = model(inputs)  # the output of model embeding
```

## 3. 模型介绍

目前支持的视觉大模型以及预训练权重如下：

|  系列  |           模型           | 模型大小 | embedding_size |                   预训练数据集                   |
| :----: | :----------------------: | :------: | :------------: | :----------------------------------------------: |
|  CLIP  |  CLIP_vit_base_patch16_224   |   85M    |      768       |                       WIT                        |
|  CLIP  |  CLIP_vit_base_patch32_224   |   87M    |      768       |                       WIT                        |
|  CLIP  |  CLIP_vit_large_patch14_224  |   302M   |      1024      |                       WIT                        |
|  CLIP  |  CLIP_vit_large_patch14_336  |   302M   |      1024      |                       WIT                        |
| BEiTv2 | BEiTv2_vit_base_patch16_224  |   85M    |      768       |                   ImageNet-1k                    |
| BEiTv2 | BEiTv2_vit_large_patch16_224 |   303M   |      1024      |                   ImageNet-1k                    |
| MoCoV3 |       MoCoV3_vit_small       |   21M    |      384       |                   ImageNet-1k                    |
| MoCoV3 |       MoCoV3_vit_base        |   85M    |      768       |                   ImageNet-1k                    |
|  MAE   |     MAE_vit_base_patch16     |   85M    |      768       |                   ImageNet-1k                    |
|  MAE   |    MAE_vit_large_patch16     |   303M   |      1024      |                   ImageNet-1k                    |
|  MAE   |     MAE_vit_huge_patch14     |   630M   |      1280      |                   ImageNet-1k                    |
|  EVA   |     EVA_vit_huge_patch14     |  1010M   |      1408      | ImageNet-21k, CC12M,   CC2M, Object365,COCO, ADE |
|  CAE   |   CAE_vit_base_patch16_224   |   85M    |      768       |                   ImageNet-1k                    |

## 4. 参考文献

1. [MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.02057.pdf)
2. [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
3. [BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/abs/2208.06366)
4. [CAE: Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/abs/2202.03026)
5. [EVA: EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://paperswithcode.com/paper/eva-exploring-the-limits-of-masked-visual)
6. [MAE: Masked Autoencoders Are Scalable Vision Learners](https://paperswithcode.com/paper/masked-autoencoders-are-scalable-vision)