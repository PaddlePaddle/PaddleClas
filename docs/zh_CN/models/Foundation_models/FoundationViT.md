# Foundation ViT介绍文档

## 目录

1. [功能介绍](#1-功能介绍)
2. [使用说明](#2-使用说明)
3. [模型介绍](#3-模型介绍)
4. [参考文献](#4-参考文献)

## 1. 功能介绍

为支持视觉大模型的使用，PaddleClas提供了各系列视觉大模型的预训练权重以及特征提取功能，可使用该功能得到在大数据上完成预训练的视觉大模型特征。

## 2. 使用说明

以模型 `CLIP_vit_base_patch16_224`为例，使用该模型以及对应的预训练权重进行特征提取的代码如下：

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

|  系列  |                模型                | 模型大小 | embedding_size |                   预训练数据集                   | 权重下载                                                                                                                         |
| :----: | :--------------------------------: | :------: | :------------: | :----------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------- |
|  CLIP  |     CLIP_vit_base_patch16_224     |   85M   |      768      |                       WIT                       | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CLIP_vit_base_patch16_224.pdparams)          |
|  CLIP  |     CLIP_vit_base_patch32_224     |   87M   |      768      |                       WIT                       | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CLIP_vit_base_patch32_224.pdparams)          |
|  CLIP  |     CLIP_vit_large_patch14_224     |   302M   |      1024      |                       WIT                       | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CLIP_vit_large_patch14_224.pdparams)         |
|  CLIP  |     CLIP_vit_large_patch14_336     |   302M   |      1024      |                       WIT                       | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CLIP_vit_large_patch14_336.pdparams)         |
| BEiTv2 |    BEiTv2_vit_base_patch16_224    |   85M   |      768      |                   ImageNet-1k                   | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/BEiTv2_vit_base_patch16_224.pdparams)        |
| BEiTv2 | BEiTv2_vit_base_patch16_224_ft21k |   85M   |      768      |            ImageNet-1k、ImageNet-21k            | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/BEiTv2_vit_base_patch16_224_ft21k.pdparams)  |
| BEiTv2 |    BEiTv2_vit_large_patch16_224    |   303M   |      1024      |                   ImageNet-1k                   | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/BEiTv2_vit_large_patch16_224.pdparams)       |
| BEiTv2 | BEiTv2_vit_large_patch16_224_ft21k |   303M   |      1024      |            ImageNet-1k、ImageNet-21k            | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/BEiTv2_vit_large_patch16_224_ft21k.pdparams) |
| MoCoV3 |          MoCoV3_vit_small          |   21M   |      384      |                   ImageNet-1k                   | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MoCoV3_vit_small.pdparams)                   |
| MoCoV3 |          MoCoV3_vit_base          |   85M   |      768      |                   ImageNet-1k                   | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MoCoV3_vit_base.pdparams)                    |
|  MAE  |        MAE_vit_base_patch16        |   85M   |      768      |                   ImageNet-1k                   | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MAE_vit_base_patch16.pdparams)               |
|  MAE  |       MAE_vit_large_patch16       |   303M   |      1024      |                   ImageNet-1k                   | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MAE_vit_large_patch16.pdparams)              |
|  MAE  |        MAE_vit_huge_patch14        |   630M   |      1280      |                   ImageNet-1k                   | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MAE_vit_huge_patch14.pdparams)               |
|  EVA  |       EVA_vit_giant_patch14       |  1010M  |      1408      | ImageNet-21k, CC12M,   CC2M, Object365,COCO, ADE | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/EVA_vit_giant_patch14.pdparams)               |
|  CAE  |      CAE_vit_base_patch16_224      |   85M   |      768      |                   ImageNet-1k                   | [下载地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CAE_vit_base_patch16_224.pdparams)           |

**备注：** PaddleClas 提供的 CLIP 系列模型在 ImageNet1k 数据集 fine-tune 的配置（[CLIP_vit_base_patch14_224](ppcls/configs/CLIP/CLIP_vit_base_patch16_224_finetune.yaml)，[CLIP_vit_large_patch14_224](ppcls/configs/CLIP/CLIP_vit_large_patch16_224_finetune.yaml)）中：
* 默认未使用 `EMA`，如需使用，请自行修改配置文件增加字段：
    ```yaml
    EMA:
      decay: 0.9999
    ```
* 数据预处理中，`NormalizeImage` 默认使用 ImageNet1k 数据集的 `mean` 和 `std` 参数（`mean` 为 `[0.485, 0.456, 0.406]`，`std` 为 `[0.229, 0.224, 0.225]`），如需使用 LAION 数据集相应参数，请自行修改相应字段：
    ```yaml
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.48145466, 0.4578275, 0.40821073]
        std: [0.26862954, 0.26130258, 0.27577711]
    ```

## 4. 参考文献

1. [MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.02057.pdf)
2. [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
3. [BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/abs/2208.06366)
4. [CAE: Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/abs/2202.03026)
5. [EVA: EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://paperswithcode.com/paper/eva-exploring-the-limits-of-masked-visual)
6. [MAE: Masked Autoencoders Are Scalable Vision Learners](https://paperswithcode.com/paper/masked-autoencoders-are-scalable-vision)
