# GhostNetV3 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 当前支持的模型](#1.2)
- [2. 模型快速体验](#2)
- [3. 模型训练、评估和预测](#3)
- [4. 模型推理部署](#4)
  - [4.1 推理模型准备](#4.1)
  - [4.2 基于 Python 预测引擎推理](#4.2)
  - [4.3 基于 C++ 预测引擎推理](#4.3)
  - [4.4 服务化部署](#4.4)
  - [4.5 端侧部署](#4.5)
  - [4.6 Paddle2ONNX 模型转换与预测](#4.6)

<a name='1'></a>

## 1. 模型介绍

<a name='1.1'></a>

### 1.1 模型简介

GhostNetV3 是华为在 GhostNet / GhostNetV2 基础上进一步提出的轻量级分类网络。

随机初始化权重前向对齐结果如下：

| Models | 对齐节点 | max abs diff | mean abs diff |
|:--:|:--:|:--:|:--:|
| GhostNetV3_x0_5 | `forward_features` | `1.60e-07` | `1.00e-08` |
| GhostNetV3_x0_5 | `out` | `1.00e-08` | `0.00e+00` |
| GhostNetV3_x1_3 | `forward_features` | `5.10e-07` | `4.00e-08` |
| GhostNetV3_x1_3 | `out` | `5.00e-08` | `1.00e-08` |
| GhostNetV3_x1_6 | `forward_features` | `5.10e-07` | `4.00e-08` |
| GhostNetV3_x1_6 | `out` | `5.00e-08` | `1.00e-08` |

对 `GhostNetV3_x1_0`，进一步完成了预训练权重对齐。使用 `timm/ghostnetv3_100.in1k` 权重转换到 Paddle 后，前向误差如下：

| Models | 权重来源 | 对齐节点 | max abs diff | mean abs diff |
|:--:|:--:|:--:|:--:|:--:|
| GhostNetV3_x1_0 | `timm/ghostnetv3_100.in1k` | `forward_features` | `2.16e-05` | `3.20e-07` |
| GhostNetV3_x1_0 | `timm/ghostnetv3_100.in1k` | `out` | `7.63e-06` | `2.66e-06` |

在完整 ImageNet1k val 上，使用同一份 `timm` 预训练权重分别在 `timm` 与 PaddleClas 中评测，结果如下：

| Models | Eval Framework | Top1 | Top5 |
|:--:|:--:|:--:|:--:|
| GhostNetV3_x1_0 | timm | `0.76930` | `0.93132` |
| GhostNetV3_x1_0 | PaddleClas | `0.76896` | `0.93132` |

进一步地，使用官方仓库 release 中提供的 `ghostnetv3-1.0.pth.tar`，分别评测其中的普通参数 `state_dict` 与指数滑动平均参数 `state_dict_ema`，结果如下：

| Models | 权重来源 | Eval Framework | Top1 | Top5 |
|:--:|:--:|:--:|:--:|:--:|
| GhostNetV3_x1_0 | official release `state_dict` | PyTorch | `0.76930` | `0.93132` |
| GhostNetV3_x1_0 | official release `state_dict_ema` | PyTorch | `0.77134` | `0.93236` |
| GhostNetV3_x1_0 | official release `state_dict_ema` | PaddleClas | `0.77080` | `0.93228` |

此外，基于 TIPC 自动准备的 lite ImageNet 数据，对 `GhostNetV3_x0_5 / x1_0 / x1_3 / x1_6` 做了 8 epoch 的快速收敛性验证。该实验仅用于验证训练链路、反向传播与优化过程是否正常，不作为全量 ImageNet 最终精度结论。训练使用统一设置：

- 数据集：`dataset/whole_chain_little_train`
- 优化器：`Momentum`
- 学习率：`0.02`
- 调度器：`Cosine`
- warmup epoch：`0`
- batch size：`8`

四个模型在训练过程中均未出现 `NaN/Inf`，训练集平均 loss 如下：

| Models | Epoch1 | Epoch2 | Epoch3 | Epoch4 | Epoch5 | Epoch6 | Epoch7 | Epoch8 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| GhostNetV3_x0_5 | `7.09630` | `6.30080` | `5.05172` | `4.55270` | `4.34154` | `4.01807` | `3.92502` | `3.71694` |
| GhostNetV3_x1_0 | `7.03280` | `6.07542` | `5.30271` | `4.51359` | `4.12840` | `3.79940` | `3.57494` | `3.39662` |
| GhostNetV3_x1_3 | `7.01563` | `5.74808` | `4.94816` | `4.15918` | `3.70600` | `3.73494` | `3.48128` | `3.28373` |
| GhostNetV3_x1_6 | `7.06830` | `5.93590` | `5.54609` | `4.65743` | `3.91060` | `3.77108` | `3.40826` | `3.37555` |

从上述结果可以看到，四个宽度变体的训练集平均 loss 都整体明显下降，说明当前 Paddle 实现已经具备正常的训练与收敛行为。

<a name='1.2'></a>

### 1.2 当前支持的模型

| Models | 权重状态 | 前向对齐 | ImageNet eval | lite ImageNet 训练链路 |
|:--:|:--:|:--:|:--:|:--:|
| GhostNetV3_x0_5 | 随机初始化 | 已验证 | 未提供预训练权重 | 已验证 |
| GhostNetV3_x1_0 | `timm` 预训练权重 | 已验证 | 已验证 | 已验证 |
| GhostNetV3_x1_3 | 随机初始化 | 已验证 | 未提供预训练权重 | 已验证 |
| GhostNetV3_x1_6 | 随机初始化 | 已验证 | 未提供预训练权重 | 已验证 |

训练配置位于：

- `ppcls/configs/ImageNet/GhostNetV3/GhostNetV3_x1_0.yaml`

TIPC 配置位于：

- `test_tipc/configs/GhostNetV3/GhostNetV3_x1_0_train_infer_python.txt`

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考 [ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

当前推荐使用的分类模型入口为：

- `GhostNetV3_x1_0`

其余宽度变体当前已完成前向对齐，但默认不提供 Paddle 侧预训练模型链接。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet 数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。`ppcls/configs/ImageNet/GhostNetV3/` 中提供了当前模型的训练与评估配置，启动方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

当前推荐优先使用：

- `GhostNetV3_x1_0.yaml`

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库，作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference 可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于 Paddle Inference 推理引擎的介绍，可以参考 [Paddle Inference 官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备)。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 Python 预测引擎推理的示例。您可以参考 [ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理)。

<a name="4.3"></a>

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考 [服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md) 来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考 [基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md) 完成相应的预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于 Paddle Serving 的介绍，可以参考 [Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考 [模型服务化部署](../../deployment/image_classification/paddle_serving.md) 来完成相应的部署工作。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考 [Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考 [端侧部署](../../deployment/image_classification/paddle_lite.md) 来完成相应的部署工作。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括 TensorRT、OpenVINO、MNN、TNN、NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考 [Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考 [Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md) 来完成相应的部署工作。
