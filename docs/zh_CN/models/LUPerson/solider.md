# Solider

-----
## 目录

- [1. 模型介绍](#1)
- [2. 对齐日志、模型](#2)

<a name='1'></a>

## 1. 模型介绍

Solider是一个语义可控的自监督学习框架，可以从大量未标记的人体图像中学习一般的人类表征，从而最大限度地有利于下游以人类为中心的任务。与已有的自监督学习方法不同，该方法利用人体图像中的先验知识建立伪语义标签，并将更多的语义信息引入到学习的表示中。同时，不同的下游任务往往需要不同比例的语义信息和外观信息，单一的学习表示不能满足所有需求。为了解决这一问题，Solider引入了一种带有语义控制器的条件网络，可以满足下游任务的不同需求。[论文地址](https://arxiv.org/abs/2303.17602)。

<a name='2'></a>

## 2. 对齐日志、模型

| model                         | weight                                                       | log                                                          |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| swin_tiny_patch4_window7_224  | https://paddleclas.bj.bcebos.com/models/SOLIDER/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams | 链接：https://pan.baidu.com/s/1W5zUFboMMhXETy4HEWbM3Q?pwd=45nx <br/>提取码：45nx |
| swin_small_patch4_window7_224 | https://paddleclas.bj.bcebos.com/models/SOLIDER/SwinTransformer_small_patch4_window7_224_pretrained.pdparams | 链接：https://pan.baidu.com/s/1sqcUdfv6FyhW9_QgxBUPWA?pwd=letv <br/>提取码：letv |
| swin_base_patch4_window7_224  | https://paddleclas.bj.bcebos.com/models/SOLIDER/SwinTransformer_base_patch4_window7_224_pretrained.pdparams | 链接：https://pan.baidu.com/s/1S2TgDxDRa72C_3FrP8duiA?pwd=u3d2 <br/>提取码：u3d2 |

[1]：基于  LUPerson 数据集预训练

<a name='3'></a>
