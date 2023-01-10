# Inception 系列
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

GoogLeNet 是 2014 年由 Google 设计的一种新的神经网络结构，其与 VGG 网络并列成为当年 ImageNet 挑战赛的双雄。GoogLeNet 首次引入 Inception 结构，在网络中堆叠该结构使得网络层数达到了 22 层，这也是卷积网络首次超过 20 层的标志。由于在 Inception 结构中使用了 1x1 的卷积用于通道数降维，并且使用了 Global-pooling 代替传统的多 fc 层加工特征的方式，最终的 GoogLeNet 网络的 FLOPs 和参数量远小于 VGG 网络，成为当时神经网络设计的一道亮丽风景线。

InceptionV3 是 Google 对 InceptionV2 的一种改进。首先，InceptionV3 对 Inception 模块进行了优化，同时设计和使用了更多种类的 Inception 模块，与此同时，InceptionV3 中的部分 Inception 模块将较大的方形二维卷积拆成两个较小的非对称卷积，这样可以大幅度节省参数量。

Xception 是 Google 继 Inception 后提出的对 InceptionV3 的另一种改进。在 Xception 中，作者使用了深度可分离卷积代替了传统的卷积操作，该操作大大节省了网络的 FLOPs 和参数量，但是精度反而有所提升。在 DeeplabV3+ 中，作者将 Xception 做了进一步的改进，同时增加了 Xception 的层数，设计出了 Xception65 和 Xception71 的网络。

InceptionV4 是 2016 年由 Google 设计的新的神经网络，当时残差结构风靡一时，但是作者认为仅使用 Inception 结构也可以达到很高的性能。InceptionV4 使用了更多的 Inception module，在 ImageNet 上的精度再创新高。

该系列模型的 FLOPs、参数量以及 T4 GPU 上的预测耗时如下图所示。

![](../../../images/models/T4_benchmark/t4.fp32.bs4.Inception.flops.png)

![](../../../images/models/T4_benchmark/t4.fp32.bs4.Inception.params.png)

![](../../../images/models/T4_benchmark/t4.fp32.bs4.Inception.png)

![](../../../images/models/T4_benchmark/t4.fp16.bs4.Inception.png)

上图反映了 Xception 系列和 InceptionV4 的精度和其他指标的关系。其中 Xception_deeplab 与论文结构保持一致，Xception 是 PaddleClas 的改进模型，在预测速度基本不变的情况下，精度提升约 0.6%。关于该改进模型的详细介绍正在持续更新中，敬请期待。

<a name='1.2'></a>

### 1.2 模型指标

| Models             | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| GoogLeNet          | 0.707  | 0.897  | 0.698             |                   | 2.880        | 8.460             |
| Xception41         | 0.793  | 0.945  | 0.790             | 0.945             | 16.740       | 22.690            |
| Xception41<br>_deeplab | 0.796  | 0.944  |                   |                   | 18.160       | 26.730            |
| Xception65         | 0.810  | 0.955  |                   |                   | 25.950       | 35.480            |
| Xception65<br>_deeplab | 0.803  | 0.945  |                   |                   | 27.370       | 39.520            |
| Xception71         | 0.811  | 0.955  |                   |                   | 31.770       | 37.280            |
| InceptionV3        | 0.791  | 0.946  | 0.788             | 0.944             | 11.460       | 23.830            |
| InceptionV4        | 0.808  | 0.953  | 0.800             | 0.950             | 24.570       | 42.680            |

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models      | Size | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
|------------------------|-------------------|------------------------|------------------------|------------------------|
| GoogLeNet              | 224       | 1.26 | 2.84 | 3.64 |
| Xception41             | 299       | 3.20 | 7.78 | 14.83 |
| Xception41_<br>deeplab | 299       | 3.34 | 8.22 | 15.54 |
| Xception65             | 299       | 5.01 | 11.66 | 22.49 |
| Xception65_<br>deeplab | 299       | 4.98 | 11.90 | 22.94 |
| Xception71             | 299       | 5.75 | 14.11 | 27.37 |
| InceptionV3            | 299       | 3.92 | 5.98 | 9.57 |
| InceptionV4            | 299       | 7.09 | 10.95 | 18.37 |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT-8.0.3.4。

<a name='1.3.2'></a>

#### 1.3.2 基于 T4 GPU 的预测速度

| Models            | Size | Latency(ms)<br>FP16<br>bs=1 | Latency(ms)<br>FP16<br>bs=4 | Latency(ms)<br>FP16<br>bs=8 | Latency(ms)<br>FP32<br>bs=1 | Latency(ms)<br>FP32<br>bs=4 | Latency(ms)<br>FP32<br>bs=8 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| GoogLeNet          | 299 | 1.75451                      | 3.39931                      | 4.71909                      | 1.88038                      | 4.48882                      | 6.94035                      |
| Xception41         | 299 | 2.91192                      | 7.86878                      | 15.53685                     | 4.96939                      | 17.01361                     | 32.67831                     |
| Xception41_<br>deeplab | 299 | 2.85934                      | 7.2075                       | 14.01406                     | 5.33541                      | 17.55938                     | 33.76232                     |
| Xception65         | 299 | 4.30126                      | 11.58371                     | 23.22213                     | 7.26158                      | 25.88778                     | 53.45426                     |
| Xception65_<br>deeplab | 299 | 4.06803                      | 9.72694                      | 19.477                       | 7.60208                      | 26.03699                     | 54.74724                     |
| Xception71         | 299 | 4.80889                      | 13.5624                      | 27.18822                     | 8.72457                      | 31.55549                     | 69.31018                     |
| InceptionV3        | 299 | 3.67502                      | 6.36071                     | 9.82645                     | 6.64054                     | 13.53630                     | 22.17355                     |
| InceptionV4        | 299 | 9.50821                      | 13.72104                     | 20.27447                     | 12.99342                     | 25.23416                     | 43.56121                     |

**备注：** 推理过程使用 TensorRT-8.0.3.4。

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/Inception/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备) 。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 python 预测引擎推理的示例。您可以参考[ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理) 。

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
