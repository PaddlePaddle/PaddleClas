# MetaCLIP

## 模型介绍

MetaCLIP 是基于 CLIP 的视觉编码器，通过元数据驱动的数据管理策略，在 CLIP 训练中实现了更好的数据质量和模型性能。

本实现为 MetaCLIP 的 PaddlePaddle 版本。

## 模型权重

| 模型 | 权重文件 | 权重地址 |
| --- | --- | --- |
| MetaCLIP_B_16 | MetaCLIP_B_16_pretrained.pdparams | [AI Studio 模型空间](https://aistudio.baidu.com/modelsdetail/51427/space) |

## 模型结构

MetaCLIP 基于 Vision Transformer (ViT) 架构，主要包含：
- Patch Embedding
- Transformer Encoder
- Layer Normalization

## 引用

```bibtex
@article{ xu2023metaclip,
  title={MetaCLIP: Everything is Data},
  author={Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Wenhan Xiong, Mike Lewis, Luke Zettlemoyer},
  year={2023},
  journal={arXiv preprint arXiv:2309.16671},
}
```
