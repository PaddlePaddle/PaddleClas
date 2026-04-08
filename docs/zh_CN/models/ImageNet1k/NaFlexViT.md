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

- 与本地 `timm` 官方预训练权重的前向对齐
- 与本地 `timm` 参考实现的随机初始化前向对齐
- 随机初始化权重转换与加载验证
- ImageNet1k 分类训练配置补充
- 静态图导出验证

前向对齐阶段使用固定输入和本地 `timm` 参考实现进行比对。当前以官方预训练权重场景为主，`forward_features` 与最终 `out` 的绝对误差均稳定低于 `1e-4`：

| Models | 权重来源 | 输入尺寸 | 对齐节点 | max abs diff | mean abs diff |
|:--:|:--:|:--:|:--:|:--:|:--:|
| `naflexvit_base_patch16_gap` | `timm` 官方预训练 | `256 x 256` | `forward_features` | `6.48e-05` | `1.40e-06` |
| `naflexvit_base_patch16_gap` | `timm` 官方预训练 | `256 x 256` | `out` | `4.29e-06` | `5.60e-07` |
| `naflexvit_base_patch16_par_gap` | `timm` 官方预训练 | `224 x 320` | `forward_features` | `7.82e-05` | `1.61e-06` |
| `naflexvit_base_patch16_par_gap` | `timm` 官方预训练 | `224 x 320` | `out` | `6.91e-06` | `7.98e-07` |
| `naflexvit_base_patch16_parfac_gap` | `timm` 官方预训练 | `224 x 320` | `forward_features` | `8.77e-05` | `1.33e-06` |
| `naflexvit_base_patch16_parfac_gap` | `timm` 官方预训练 | `224 x 320` | `out` | `8.58e-06` | `6.37e-07` |

说明：

- 上表结果在 `torch=cuda`、`paddle=gpu` 的双端 GPU 环境下获得
- `par_gap` 变体依赖 Paddle GPU 上的 `bicubic + antialias` 插值核进行高精度对齐
- 当前阶段不提供全量 ImageNet 训练精度与 Paddle 预训练权重下载链接
- 随机初始化权重场景也已验证，当前 3 个基础变体的 `forward_features max abs diff` 均在 `1e-6` 量级

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
- `ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_gap_lite_imagenet.yaml`

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
python tools/verify_naflexvit_alignment.py --variant naflexvit_base_patch16_gap --height 256 --width 256 --batch-size 2 --pretrained --torch-python /root/timm_env_cu126/bin/python --paddle-python /root/paddleclas_env/bin/python --torch-device cuda --paddle-device gpu
python tools/verify_naflexvit_alignment.py --variant naflexvit_base_patch16_par_gap --height 224 --width 320 --batch-size 2 --pretrained --torch-python /root/timm_env_cu126/bin/python --paddle-python /root/paddleclas_env/bin/python --torch-device cuda --paddle-device gpu
python tools/verify_naflexvit_alignment.py --variant naflexvit_base_patch16_parfac_gap --height 224 --width 320 --batch-size 2 --pretrained --torch-python /root/timm_env_cu126/bin/python --paddle-python /root/paddleclas_env/bin/python --torch-device cuda --paddle-device gpu
```

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet 数据准备、模型训练、评估和预测等内容。`ppcls/configs/ImageNet/NaFlexViT/` 中已经提供该系列分类模型的训练配置，启动方式可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

当前推荐优先使用以下配置：

- `naflexvit_base_patch16_gap.yaml`
- `naflexvit_base_patch16_par_gap.yaml`
- `naflexvit_base_patch16_parfac_gap.yaml`
- `naflexvit_base_patch16_gap_lite_imagenet.yaml`

若只需要做小数据快速收敛验证，可直接复用 TIPC 的 `lite_train_lite_infer` 数据准备流程：

```bash
bash test_tipc/prepare.sh test_tipc/configs/NaFlexViT/naflexvit_base_patch16_gap_train_infer_python.txt lite_train_lite_infer
```

该命令会自动下载并准备 `dataset/whole_chain_little_train`，同时建立 `dataset/ILSVRC2012` 软链接和对应的 `train_list.txt`、`val_list.txt`，无需额外准备全量 ImageNet。

在此基础上，可直接用 GPU 跑一个短周期收敛实验，例如：

```bash
python tools/train.py -c ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_gap_lite_imagenet.yaml -o Global.device=gpu
```

本地已完成 1 次 5 epoch 的 GPU 收敛性验证，环境为 `PaddlePaddle 3.3.0 + A100`，结果如下：

| 配置 | 数据 | 设备 | Epoch | Train CELoss | Train Top1 | Train Top5 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `naflexvit_base_patch16_gap_lite_imagenet.yaml` | TIPC `lite_train_lite_infer` | `gpu:0` | 1 | `7.20865` | `0.00000` | `0.00000` |
| `naflexvit_base_patch16_gap_lite_imagenet.yaml` | TIPC `lite_train_lite_infer` | `gpu:0` | 5 | `3.46303` | `0.20690` | `0.58621` |

从训练集指标看，loss 明显下降，Top-1 / Top-5 持续上升，可作为训练链路能够正常收敛的快速验证。本次实验仅用于证明收敛，不作为全量 ImageNet 精度结论。

<a name="4"></a>

## 4. 当前实验结论与已知限制

当前已经完成的实验：

- 3 个基础变体的 `timm` 官方预训练权重前向对齐
- 3 个基础变体的随机初始化前向对齐
- `naflexvit_base_patch16_gap` 在 TIPC 小数据集上的 5 epoch GPU 收敛性验证
- 随机初始化权重转换与加载验证
- 配置文件补充与可实例化验证
- 3 个基础变体的静态图导出验证

当前尚未纳入本次提交结论的内容：

- 全量 ImageNet 精度指标
- Paddle 预训练权重下载链接

静态图导出验证命令示例：

```bash
python tools/export_model.py -c ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_gap.yaml -o Global.save_inference_dir=./inference/naflexvit_base_patch16_gap
python tools/export_model.py -c ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_par_gap.yaml -o Global.save_inference_dir=./inference/naflexvit_base_patch16_par_gap
python tools/export_model.py -c ppcls/configs/ImageNet/NaFlexViT/naflexvit_base_patch16_parfac_gap.yaml -o Global.save_inference_dir=./inference/naflexvit_base_patch16_parfac_gap
```

在 Paddle 3.3 环境下，上述 3 个基础变体已经完成导出验证并可成功生成 `inference.pdmodel` 与 `inference.pdiparams`。
