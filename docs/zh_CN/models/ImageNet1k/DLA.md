# DLA 系列
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

DLA(Deep Layer Aggregation)。 视觉识别需要丰富的表示形式，其范围从低到高，范围从小到大，分辨率从精细到粗糙。即使卷积网络中的要素深度很深，仅靠隔离层还是不够的：将这些表示法进行复合和聚合可改善对内容和位置的推断。尽管已合并了残差连接以组合各层，但是这些连接本身是“浅”的，并且只能通过简单的一步操作来融合。作者通过更深层的聚合来增强标准体系结构，以更好地融合各层的信息。Deep Layer Aggregation 结构迭代地和分层地合并了特征层次结构，以使网络具有更高的准确性和更少的参数。跨体系结构和任务的实验表明，与现有的分支和合并方案相比，Deep Layer Aggregation 可提高识别和分辨率。[论文地址](https://arxiv.org/abs/1707.06484)。

<a name='1.2'></a>

### 1.2 模型指标
| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DLA34                 | 76.03     |   92.98   | - | - | 3.1  | 15.8   |  
| DLA46_c               | 63.21     |   85.30   | - | - | 0.5  | 1.3    |  
| DLA46x_c              | 64.36     |   86.01   | - | - | 0.5  | 1.1    |  
| DLA60                 | 76.10     |   92.92   | - | - | 4.2  | 22.0   |  
| DLA60x                | 77.53     |   93.78   | - | - | 3.5  | 17.4   |  
| DLA60x_c              | 66.45     |   87.54   | - | - | 0.6  | 1.3    |  
| DLA102                | 78.93     |   94.52   | - | - | 7.2  | 33.3   |  
| DLA102x               | 78.10     |   94.00   | - | - | 5.9  | 26.4   |  
| DLA102x2              | 78.85     |   94.45   | - | - | 9.3  | 41.4   |  
| DLA169                | 78.09     |   94.09   | - | - | 11.6 | 53.5   |  

**备注：** PaddleClas 所提供的该系列模型的预训练模型权重，均是基于其官方提供的权重转得。

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models     | Size  | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
| -------- | --------- | ----------------- | ------------------------------ | ------------------------------ |
| DLA102   | 224       | 4.15                           | 6.81                           | 11.60                          |
| DLA102x2 | 224       | 6.40                          | 16.08                          | 33.51                          |
| DLA102x  | 224       | 4.68                          | 16.44                          | 20.98                          |
| DLA169   | 224       | 6.45                           | 10.79                          | 18.31                          |
| DLA34    | 224       | 1.67                           | 2.49                           | 4.31                           |
| DLA46_c  | 224       | 0.88                           | 1.44                           | 1.96                           |
| DLA60    | 224       | 2.54                           | 4.26                           | 7.01                           |
| DLA60x_c | 224       | 1.04                           | 1.82                           | 3.68                           |
| DLA60x   | 224       | 2.66                           | 8.44                           | 11.95                          |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT-8.0.3.4。

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/DLA/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

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
