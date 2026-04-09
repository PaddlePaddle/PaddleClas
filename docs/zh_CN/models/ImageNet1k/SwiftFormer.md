# SwiftFormer 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
    - [1.3 Benchmark](#1.3)
          - [1.3.1 基于 V100 GPU 的预测速度](#1.3.1)
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

SwiftFormer 于2023年提出，聚焦于解决视觉 Transformer 在移动端部署中，全局自注意力计算开销大、推理延迟高的问题。它认为，限制轻量级视觉模型效率的关键瓶颈，不仅在于参数量或 FLOPs，更在于注意力模块在实际硬件上的访问与计算代价。为此，SwiftFormer 重新审视了高效视觉建模中的 token mixing 方式，提出了 Efficient Additive Attention，用加性建模替代传统点积自注意力，在保留全局信息交互能力的同时，显著降低了计算复杂度与部署延迟。基于这一设计，进一步提出了 SwiftFormer，一个面向移动视觉任务的简洁高效骨干网络。该模型在紧凑的网络结构和有限的计算预算下，实现了优异的精度—速度权衡，兼具较强的表示能力与实际部署效率。[论文地址：https://arxiv.org/abs/2303.15446]

当前 PaddleClas 已支持以下 SwiftFormer 变体：

- SwiftFormer_XS
- SwiftFormer_S
- SwiftFormer_L1
- SwiftFormer_L3

对应配置位于：`ppcls/configs/ImageNet/SwiftFormer/`。

<a name='1.2'></a>

### 1.2 模型指标

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SwiftFormer_XS    | 0.7561 | 0.9238 | 0.757 | - | 0.611 | 3.5 |
| SwiftFormer_S     | 0.7841 | 0.9393 | 0.785 | - | 0.995 | 6.1 |
| SwiftFormer_L1    | 0.8091 | 0.9528 | 0.809 | - | 1.609 | 12.1 |
| SwiftFormer_L3    | 0.8300 | 0.9617 | 0.830 | - | 4.029 | 28.5 |

**备注：** PaddleClas 所提供的该系列模型的预训练模型权重，均是基于其官方提供的权重转得。

<a name='1.3'></a>

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models      | Size | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
|:--:|:--:|:--:|:--:|:--:|
| SwiftFormer_XS    | ** | ** | ** | ** |
| SwiftFormer_S    | ** | ** | ** | ** |
| SwiftFormer_L1    | ** | ** | ** | ** |
| SwiftFormer_L3    | ** | ** | ** | ** |

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet 数据准备、该模型在 ImageNet 上的训练、评估、预测等内容。

在 `ppcls/configs/ImageNet/SwiftFormer/` 中提供了该模型配置文件。

示例：

```bash
# 训练
python3 -m paddle.distributed.launch     --gpus="0"     tools/train.py -c /home/housaijie/code/PaddleClas/ppcls/configs/ImageNet/SwiftFormer/SwiftFormer_L1.yaml



# 评估（以本地权重为例）
python tools/eval.py -c ppcls/configs/ImageNet/SwiftFormer/SwiftFormer_L1.yaml \
  -o Global.pretrained_model=/path/to/SwiftFormer_L1.pdparams
```

其他训练、评估、预测通用流程可参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库，作用于服务器端和云端，提供高性能推理能力。更多关于 Paddle Inference 的介绍可参考 [Paddle Inference 官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备)。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 Python 预测引擎推理的示例。您可以参考 [ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理)。

<a name="4.3"></a>

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md)完成相应推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md)完成预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。更多介绍可参考 [Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 的服务化部署示例，您可以参考[模型服务化部署](../../deployment/image_classification/paddle_serving.md)完成部署。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是轻量级高性能推理框架，支持移动端、嵌入式和服务器端多硬件平台。更多介绍可参考 [Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 的端侧部署示例，您可以参考[端侧部署](../../deployment/image_classification/paddle_lite.md)完成部署。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 Paddle 模型格式转换为 ONNX 格式，便于在 TensorRT/OpenVINO/NCNN 等推理引擎中部署。更多介绍可参考 [Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)完成转换与预测。
