# ResNeXt 系列
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

ResNeXt 是 ResNet 的典型变种网络之一，ResNeXt 发表于 2017 年的 CVPR 会议。在此之前，提升模型精度的方法主要集中在将网络变深或者变宽，这样增加了参数量和计算量，推理速度也会相应变慢。ResNeXt 结构提出了通道分组（cardinality）的概念，作者通过实验发现增加通道的组数比增加深度和宽度更有效。其可以在不增加参数复杂度的前提下提高准确率，同时还减少了参数的数量，所以是比较成功的 ResNet 的变种。

该系列模型的 FLOPs、参数量以及 T4 GPU 上的预测耗时如下图所示。

![](../../images/models/T4_benchmark/t4.fp32.bs4.SeResNeXt.flops.png)

![](../../images/models/T4_benchmark/t4.fp32.bs4.SeResNeXt.params.png)

![](../../images/models/T4_benchmark/t4.fp32.bs4.SeResNeXt.png)

![](../../images/models/T4_benchmark/t4.fp16.bs4.SeResNeXt.png)

目前 PaddleClas 开源的 ResNeXt 相关预训练模型一共有 15 个，其指标如图所示，从图中可以看出，在同样 FLOPs 和 Params 下，改进版的模型往往有更高的精度，但是推理速度往往不如 ResNet 系列。

<a name='1.2'></a>

### 1.2 模型指标

| Models                | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeXt50_32x4d       | 0.778  | 0.938  | 0.778             |                   | 8.020        | 23.640            |
| ResNeXt50_vd_32x4d    | 0.796  | 0.946  |                   |                   | 8.500        | 23.660            |
| ResNeXt50_64x4d       | 0.784  | 0.941  |                   |                   | 15.060       | 42.360            |
| ResNeXt50_vd_64x4d    | 0.801  | 0.949  |                   |                   | 15.540       | 42.380            |
| ResNeXt101_32x4d      | 0.787  | 0.942  | 0.788             |                   | 15.010       | 41.540            |
| ResNeXt101_vd_32x4d   | 0.803  | 0.951  |                   |                   | 15.490       | 41.560            |
| ResNeXt101_64x4d      | 0.784  | 0.945  | 0.796             |                   | 29.050       | 78.120            |
| ResNeXt101_vd_64x4d   | 0.808  | 0.952  |                   |                   | 29.530       | 78.140            |
| ResNeXt152_32x4d      | 0.790  | 0.943  |                   |                   | 22.010       | 56.280            |
| ResNeXt152_vd_32x4d   | 0.807  | 0.952  |                   |                   | 22.490       | 56.300            |
| ResNeXt152_64x4d      | 0.795  | 0.947  |                   |                   | 43.030       | 107.570           |
| ResNeXt152_vd_64x4d   | 0.811  | 0.953  |                   |                   | 43.520       | 107.590           |

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models      | Size | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
|-----------------------|-------------------|-----------------------|-----------------------|-----------------------|
| ResNeXt50_32x4d       | 224       | 2.42              | 8.42              | 11.54             |
| ResNeXt50_vd_32x4d    | 224       | 2.50               | 8.62               | 11.90              |
| ResNeXt50_64x4d       | 224       | 3.62              | 10.24             | 20.93             |
| ResNeXt50_vd_64x4d    | 224       | 3.68              | 10.30             | 21.20             |
| ResNeXt101_32x4d      | 224       | 4.81             | 17.60             | 22.98             |
| ResNeXt101_vd_32x4d   | 224       | 4.85             | 17.50             | 23.11             |
| ResNeXt101_64x4d      | 224       | 7.12             | 20.17             | 41.64             |
| ResNeXt101_vd_64x4d   | 224       | 7.34             | 22.46             | 41.79             |
| ResNeXt152_32x4d      | 224       | 7.09             | 27.16             | 34.32             |
| ResNeXt152_vd_32x4d   | 224       | 7.12             | 26.83             | 34.48             |
| ResNeXt152_64x4d      | 224       | 10.88             | 30.14             | 62.60             |
| ResNeXt152_vd_64x4d   | 224       | 10.58             | 30.30             | 62.94             |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT-8.0.3.4。

<a name='1.3.2'></a>

#### 1.3.2 基于 T4 GPU 的预测速度

| Models            | Size | Latency(ms)<br>FP16<br>bs=1 | Latency(ms)<br>FP16<br>bs=4 | Latency(ms)<br>FP16<br>bs=8 | Latency(ms)<br>FP32<br>bs=1 | Latency(ms)<br>FP32<br>bs=4 | Latency(ms)<br>FP32<br>bs=8 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeXt50_32x4d       | 224  | 7.61087                      | 8.88918                      | 12.99674                     | 7.56327                      | 10.6134                      | 18.46915                     |
| ResNeXt50_vd_32x4d    | 224  | 7.69065                      | 8.94014                      | 13.4088                      | 7.62044                      | 11.03385                     | 19.15339                     |
| ResNeXt50_64x4d       | 224  | 13.78688                     | 15.84655                     | 21.79537                     | 13.80962                     | 18.4712                      | 33.49843                     |
| ResNeXt50_vd_64x4d    | 224  | 13.79538                     | 15.22201                     | 22.27045                     | 13.94449                     | 18.88759                     | 34.28889                     |
| ResNeXt101_32x4d      | 224  | 16.59777                     | 17.93153                     | 21.36541                     | 16.21503                     | 19.96568                     | 33.76831                     |
| ResNeXt101_vd_32x4d   | 224  | 16.36909                     | 17.45681                     | 22.10216                     | 16.28103                     | 20.25611                     | 34.37152                     |
| ResNeXt101_64x4d      | 224  | 30.12355                     | 32.46823                     | 38.41901                     | 30.4788                      | 36.29801                     | 68.85559                     |
| ResNeXt101_vd_64x4d   | 224  | 30.34022                     | 32.27869                     | 38.72523                     | 30.40456                     | 36.77324                     | 69.66021                     |
| ResNeXt152_32x4d      | 224  | 25.26417                     | 26.57001                     | 30.67834                     | 24.86299                     | 29.36764                     | 52.09426                     |
| ResNeXt152_vd_32x4d   | 224  | 25.11196                     | 26.70515                     | 31.72636                     | 25.03258                     | 30.08987                     | 52.64429                     |
| ResNeXt152_64x4d      | 224  | 46.58293                     | 48.34563                     | 56.97961                     | 46.7564                      | 56.34108                     | 106.11736                    |
| ResNeXt152_vd_64x4d   | 224  | 47.68447                     | 48.91406                     | 57.29329                     | 47.18638                     | 57.16257                     | 107.26288                    |

**备注：** 推理过程使用 TensorRT-8.0.3.4。

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/ResNeXt/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

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
