# NaFlexViT 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 当前支持的模型](#1.2)
- [2. 模型快速体验](#2)
- [3. 模型训练、评估和预测](#3)
- [4. 当前实验结论与已知限制](#4)

<a name='1'></a>

## 1. 模型介绍

<a name='1.1'></a>

### 1.1 模型简介

NaFlexViT 是 `timm` 中面向灵活输入场景实现的一类 Vision Transformer 结构，核心目标是在保持 ViT 主体结构的同时，支持：

- 基于线性 patch embedding 的 patch 表示
- learned / factorized 位置编码
- 变长宽输入下的位置编码插值
- register tokens 与 global average pooling

当前 PaddleClas 中提供的实现以本地 `timm` 的 `naflexvit.py` 为参考，优先覆盖分类主干与前向对齐所需的最小能力集合。当前已完成以下核验工作：

- 与本地 `timm` 参考实现的同权重前向对齐
- 随机初始化权重转换与加载验证
- ImageNet1k 分类训练配置补充
- 静态图导出验证

前向对齐阶段使用固定输入、同源随机初始化权重和本地 `timm` 参考实现进行比对。当前已经完成 3 个基础变体的对齐验证，`forward_features` 与最终 `out` 的绝对误差均稳定低于 `1e-4`：

| Models | 输入尺寸 | 对齐节点 | max abs diff | mean abs diff |
|:--:|:--:|:--:|:--:|:--:|
| `naflexvit_base_patch16_gap` | `256 x 256` | `forward_features` | `2.03e-06` | `1.71e-07` |
| `naflexvit_base_patch16_gap` | `256 x 256` | `out` | `1.19e-06` | `2.47e-07` |
| `naflexvit_base_patch16_par_gap` | `224 x 320` | `forward_features` | `2.38e-06` | `1.71e-07` |
| `naflexvit_base_patch16_par_gap` | `224 x 320` | `out` | `9.54e-07` | `2.40e-07` |
| `naflexvit_base_patch16_parfac_gap` | `224 x 320` | `forward_features` | `2.38e-06` | `1.71e-07` |
| `naflexvit_base_patch16_parfac_gap` | `224 x 320` | `out` | `1.13e-06` | `2.34e-07` |

说明：

- 对齐脚本默认使用 `torch=cpu`、`paddle=gpu` 的环境组合
- `par_gap` 变体依赖 Paddle GPU 上的 `bicubic + antialias` 插值核进行高精度对齐
- 当前阶段不提供全量 ImageNet 训练精度与预训练权重下载链接

<a name='1.2'></a>

### 1.2 当前支持的模型

| Models | 位置编码 | 用途 | 输出形式 | 训练配置 | 前向对齐 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| `naflexvit_base_patch16_gap` | learned | ImageNet1k 分类 | logits | 支持 | 已验证 |
| `naflexvit_base_patch16_par_gap` | learned + aspect preserving | ImageNet1k 分类 | logits | 支持 | 已验证 |
| `naflexvit_base_patch16_parfac_gap` | factorized + aspect preserving | ImageNet1k 分类 | logits | 支持 | 已验证 |

训练配置位于：

- `ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_gap.yaml`
- `ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_par_gap.yaml`
- `ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_parfac_gap.yaml`

前向对齐脚本位于：

- `tools/verify_naflexvit_alignment.py`

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 后，可以参考 [ResNet50 模型快速体验](./ResNet.md#2-模型快速体验) 进行预测。使用 NaFlexViT 时，可将 `Arch.name` 设置为：

- `naflexvit_base_patch16_gap`
- `naflexvit_base_patch16_par_gap`
- `naflexvit_base_patch16_parfac_gap`

若需要复现实验中的前向对齐，可直接运行：

```bash
python tools/verify_naflexvit_alignment.py --variant naflexvit_base_patch16_gap --height 256 --width 256 --batch-size 2
python tools/verify_naflexvit_alignment.py --variant naflexvit_base_patch16_par_gap --height 224 --width 320 --batch-size 2
python tools/verify_naflexvit_alignment.py --variant naflexvit_base_patch16_parfac_gap --height 224 --width 320 --batch-size 2
```

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet 数据准备、模型训练、评估和预测等内容。`ppcls/configs/ImageNet/NaFlexViT/` 中已经提供该系列分类模型的训练配置，启动方式可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

当前推荐优先使用以下配置：

- `naflexvit_base_patch16_gap.yaml`
- `naflexvit_base_patch16_par_gap.yaml`
- `naflexvit_base_patch16_parfac_gap.yaml`

由于当前仓库环境下未提供 ImageNet 或 lite ImageNet 数据，本次提交阶段未补充训练收敛性日志表。若后续补齐数据，可直接基于上述配置继续完成训练链路验证。

<a name="4"></a>

## 4. 当前实验结论与已知限制

当前已经完成的实验：

- 3 个基础变体的同权重前向对齐
- 随机初始化权重转换与加载验证
- 配置文件补充与可实例化验证
- 3 个基础变体的静态图导出验证

当前尚未纳入本次提交结论的内容：

- 全量 ImageNet 精度指标
- lite ImageNet 快速收敛性实验

静态图导出验证命令示例：

```bash
python tools/export_model.py -c ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_gap.yaml -o Global.save_inference_dir=./inference/naflexvit_base_patch16_gap
python tools/export_model.py -c ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_par_gap.yaml -o Global.save_inference_dir=./inference/naflexvit_base_patch16_par_gap
python tools/export_model.py -c ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_parfac_gap.yaml -o Global.save_inference_dir=./inference/naflexvit_base_patch16_parfac_gap
```

在 Paddle 3.3 环境下，上述 3 个基础变体已经完成导出验证并可成功生成 `inference.pdmodel` 与 `inference.pdiparams`。
