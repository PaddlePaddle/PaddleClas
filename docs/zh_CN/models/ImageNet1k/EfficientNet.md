# EfficientNet 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
    - [1.3 Benchmark](#1.3)
      - [1.3.1 基于 V100 GPU 的预测速度](#1.3.1)
      - [1.3.2 基于 T4 GPU 的预测速度](#1.3.2)
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

EfficientNet 是 Google 于 2019 年发布的一个基于 NAS 的轻量级网络，其中 EfficientNetB7 刷新了当时 ImageNet-1k 的分类准确率。在该文章中，作者指出，传统的提升神经网络性能的方法主要是从网络的宽度、网络的深度、以及输入图片的分辨率入手，但是作者通过实验发现，平衡这三个维度对精度和效率的提升至关重要，于是，作者通过一系列的实验中总结出了如何同时平衡这三个维度的放缩，与此同时，基于这种放缩方法，作者在 EfficientNet_B0 的基础上，构建了 EfficientNet 系列中 B1-B7 共 7 个网络，并在同样 FLOPs 与参数量的情况下，精度达到了 state-of-the-art 的效果。

该系列模型的 FLOPs、参数量以及 T4 GPU 上的预测耗时如下图所示。

![](../../images/models/T4_benchmark/t4.fp32.bs4.EfficientNet.flops.png)

![](../../images/models/T4_benchmark/t4.fp32.bs4.EfficientNet.params.png)

![](../../images/models/T4_benchmark/t4.fp32.bs1.EfficientNet.png)

![](../../images/models/T4_benchmark/t4.fp16.bs1.EfficientNet.png)

目前 PaddleClas 开源的 EfficientNet 与 ResNeXt 预训练模型一共有 14 个。从上图中可以看出 EfficientNet 系列网络优势非常明显，EfficientNet_B0_Small 是去掉了 SE_block 的 EfficientNet_B0，其具有更快的推理速度。

<a name='1.2'></a>

### 1.2 模型指标

| Models                        | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| EfficientNetB0                | 0.774  | 0.933  | 0.773             | 0.935             | 0.720        | 5.100             |
| EfficientNetB1                | 0.792  | 0.944  | 0.792             | 0.945             | 1.270        | 7.520             |
| EfficientNetB2                | 0.799  | 0.947  | 0.803             | 0.950             | 1.850        | 8.810             |
| EfficientNetB3                | 0.812  | 0.954  | 0.817             | 0.956             | 3.430        | 11.840            |
| EfficientNetB4                | 0.829  | 0.962  | 0.830             | 0.963             | 8.290        | 18.760            |
| EfficientNetB5                | 0.836  | 0.967  | 0.837             | 0.967             | 19.510       | 29.610            |
| EfficientNetB6                | 0.840  | 0.969  | 0.842             | 0.968             | 36.270       | 42.000            |
| EfficientNetB7                | 0.843  | 0.969  | 0.844             | 0.971             | 72.350       | 64.920            |
| EfficientNetB0_<br>small      | 0.758  | 0.926  |                   |                   | 0.720        | 4.650             |

**备注：** PaddleClas 所提供的该系列模型中，EfficientNetB1-B7模型的预训练模型权重，均是基于其官方提供的权重转得。

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models                               | Size | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
|-------------------------------|-------------------|-------------------------------|-------------------------------|-------------------------------|
| EfficientNetB0                | 224       | 1.58 | 2.55 | 3.69 |
| EfficientNetB1                | 240       | 2.29 | 3.92 | 5.50 |
| EfficientNetB2                | 260       | 2.52 | 4.47 | 6.78 |
| EfficientNetB3                | 300       | 3.44 | 6.53 | 10.44 |
| EfficientNetB4                | 380       | 5.35 | 11.69 | 19.97 |
| EfficientNetB5                | 456       | 8.52 | 21.94 | 38.37 |
| EfficientNetB6                | 528       | 13.49 | 36.99 | 67.17 |
| EfficientNetB7                | 600       | 21.91 | 62.29 | 116.07 |
| EfficientNetB0_<br>small      | 224       | 1.24 | 2.59 | 3.92 |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT-8.0.3.4。

<a name='1.3.2'></a>

#### 1.3.2 基于 T4 GPU 的预测速度

| Models            | Size | Latency(ms)<br>FP16<br>bs=1 | Latency(ms)<br>FP16<br>bs=4 | Latency(ms)<br>FP16<br>bs=8 | Latency(ms)<br>FP32<br>bs=1 | Latency(ms)<br>FP32<br>bs=4 | Latency(ms)<br>FP32<br>bs=8 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| EfficientNetB0            | 224      | 3.40122                      | 5.95851                      | 9.10801                      | 3.442                        | 6.11476                      | 9.3304                       |
| EfficientNetB1            | 240      | 5.25172                      | 9.10233                      | 14.11319                     | 5.3322                       | 9.41795                      | 14.60388                     |
| EfficientNetB2            | 260      | 5.91052                      | 10.5898                      | 17.38106                     | 6.29351                      | 10.95702                     | 17.75308                     |
| EfficientNetB3            | 300      | 7.69582                      | 16.02548                     | 27.4447                      | 7.67749                      | 16.53288                     | 28.5939                      |
| EfficientNetB4            | 380      | 11.55585                     | 29.44261                     | 53.97363                     | 12.15894                     | 30.94567                     | 57.38511                     |
| EfficientNetB5            | 456      | 19.63083                     | 56.52299                     | -                            | 20.48571                     | 61.60252                     | -                            |
| EfficientNetB6            | 528      | 30.05911                     | -                            | -                            | 32.62402                     | -                            | -                            |
| EfficientNetB7            | 600      | 47.86087                     | -                            | -                            | 53.93823                     | -                            | -                            |
| EfficientNetB0_small      | 224      | 2.39166                      | 4.36748                      | 6.96002                      | 2.3076                       | 4.71886                      | 7.21888                      |

**备注：** 推理过程使用 TensorRT-8.0.3.4。

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/EfficientNet/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#4.1) 。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 python 预测引擎推理的示例。您可以参考[ResNet50 基于 Python 预测引擎推理](./ResNet.md#4.2) 完成模型的推理预测。

<a name="4.3"></a>

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md)完成相应的预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../../deployment/image_classification/paddle_serving.md)来完成相应的部署工作。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../../deployment/image_classification/paddle_lite.md)来完成相应的部署工作。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)来完成相应的部署工作。
