# EfficientNet 与 ResNeXt101_wsl 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
    - [1.3 Benchmark](#1.3)
      - [1.3.1 基于 V100 GPU 的预测速度](#1.3.1)
      - [1.3.2 基于 V100 GPU 的预测速度](#1.3.2)
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

EfficientNet 是 Google 于 2019 年发布的一个基于 NAS 的轻量级网络，其中 EfficientNetB7 刷新了当时 ImageNet-1k 的分类准确率。在该文章中，作者指出，传统的提升神经网络性能的方法主要是从网络的宽度、网络的深度、以及输入图片的分辨率入手，但是作者通过实验发现，平衡这三个维度对精度和效率的提升至关重要，于是，作者通过一系列的实验中总结出了如何同时平衡这三个维度的放缩，与此同时，基于这种放缩方法，作者在 EfficientNet_B0 的基础上，构建了 EfficientNet 系列中 B1-B7 共 7 个网络，并在同样 FLOPS 与参数量的情况下，精度达到了 state-of-the-art 的效果。

ResNeXt 是 facebook 于 2016 年提出的一种对 ResNet 的改进版网络。在 2019 年，facebook 通过弱监督学习研究了该系列网络在 ImageNet 上的精度上限，为了区别之前的 ResNeXt 网络，该系列网络的后缀为 wsl，其中 wsl 是弱监督学习（weakly-supervised-learning）的简称。为了能有更强的特征提取能力，研究者将其网络宽度进一步放大，其中最大的 ResNeXt101_32x48d_wsl 拥有 8 亿个参数，将其在 9.4 亿的弱标签图片下训练并在 ImageNet-1k 上做 finetune，最终在 ImageNet-1k 的 top-1 达到了 85.4%，这也是迄今为止在 ImageNet-1k 的数据集上以 224x224 的分辨率下精度最高的网络。Fix-ResNeXt 中，作者使用了更大的图像分辨率，针对训练图片和验证图片数据预处理不一致的情况下做了专门的 Fix 策略，并使得 ResNeXt101_32x48d_wsl 拥有了更高的精度，由于其用到了 Fix 策略，故命名为 Fix-ResNeXt101_32x48d_wsl。


该系列模型的 FLOPS、参数量以及 T4 GPU 上的预测耗时如下图所示。

![](../../../images/models/T4_benchmark/t4.fp32.bs4.EfficientNet.flops.png)

![](../../../images/models/T4_benchmark/t4.fp32.bs4.EfficientNet.params.png)

![](../../../images/models/T4_benchmark/t4.fp32.bs1.EfficientNet.png)

![](../../../images/models/T4_benchmark/t4.fp16.bs1.EfficientNet.png)

目前 PaddleClas 开源的这两类模型的预训练模型一共有 14 个。从上图中可以看出 EfficientNet 系列网络优势非常明显，ResNeXt101_wsl 系列模型由于用到了更多的数据，最终的精度也更高。EfficientNet_B0_Small 是去掉了 SE_block 的 EfficientNet_B0，其具有更快的推理速度。

<a name='1.2'></a>

### 1.2 模型指标

| Models                        | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeXt101_<br>32x8d_wsl      | 0.826  | 0.967  | 0.822             | 0.964             | 29.140       | 78.440            |
| ResNeXt101_<br>32x16d_wsl     | 0.842  | 0.973  | 0.842             | 0.972             | 57.550       | 152.660           |
| ResNeXt101_<br>32x32d_wsl     | 0.850  | 0.976  | 0.851             | 0.975             | 115.170      | 303.110           |
| ResNeXt101_<br>32x48d_wsl     | 0.854  | 0.977  | 0.854             | 0.976             | 173.580      | 456.200           |
| Fix_ResNeXt101_<br>32x48d_wsl | 0.863  | 0.980  | 0.864             | 0.980             | 354.230      | 456.200           |
| EfficientNetB0                | 0.774  | 0.933  | 0.773             | 0.935             | 0.720        | 5.100             |
| EfficientNetB1                | 0.792  | 0.944  | 0.792             | 0.945             | 1.270        | 7.520             |
| EfficientNetB2                | 0.799  | 0.947  | 0.803             | 0.950             | 1.850        | 8.810             |
| EfficientNetB3                | 0.812  | 0.954  | 0.817             | 0.956             | 3.430        | 11.840            |
| EfficientNetB4                | 0.829  | 0.962  | 0.830             | 0.963             | 8.290        | 18.760            |
| EfficientNetB5                | 0.836  | 0.967  | 0.837             | 0.967             | 19.510       | 29.610            |
| EfficientNetB6                | 0.840  | 0.969  | 0.842             | 0.968             | 36.270       | 42.000            |
| EfficientNetB7                | 0.843  | 0.969  | 0.844             | 0.971             | 72.350       | 64.920            |
| EfficientNetB0_<br>small      | 0.758  | 0.926  |                   |                   | 0.720        | 4.650             |

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models                               | Crop Size | Resize Short Size | FP32<br/>Batch Size=1<br/>(ms) | FP32<br/>Batch Size=4<br/>(ms) | FP32<br/>Batch Size=8<br/>(ms) |
|-------------------------------|-----------|-------------------|-------------------------------|-------------------------------|-------------------------------|
| ResNeXt101_<br>32x8d_wsl      | 224       | 256               | 13.55 | 23.39 | 36.18 |
| ResNeXt101_<br>32x16d_wsl     | 224       | 256               | 21.96 | 38.35 | 63.29 |
| ResNeXt101_<br>32x32d_wsl     | 224       | 256               | 37.28 | 76.50 | 121.56 |
| ResNeXt101_<br>32x48d_wsl     | 224       | 256               | 55.07 | 124.39 | 205.01 |
| Fix_ResNeXt101_<br>32x48d_wsl | 320       | 320               | 55.01 | 122.63 | 204.66 |
| EfficientNetB0                | 224       | 256               | 1.96 | 3.71 | 5.56 |
| EfficientNetB1                | 240       | 272               | 2.88 | 5.40 | 7.63 |
| EfficientNetB2                | 260       | 292               | 3.26 | 6.20 | 9.17 |
| EfficientNetB3                | 300       | 332               | 4.52 | 8.85 | 13.54 |
| EfficientNetB4                | 380       | 412               | 6.78 | 15.47 | 24.95 |
| EfficientNetB5                | 456       | 488               | 10.97 | 27.24 | 45.93 |
| EfficientNetB6                | 528       | 560               | 17.09 | 43.32 | 76.90 |
| EfficientNetB7                | 600       | 632               | 25.91 | 71.23 | 128.20 |
| EfficientNetB0_<br>small      | 224       | 256               | 1.24 | 2.59 | 3.92 |

<a name='1.3.2'></a>

## 1.3.2 基于 T4 GPU 的预测速度

| Models                    | Crop Size | Resize Short Size | FP16<br>Batch Size=1<br>(ms) | FP16<br>Batch Size=4<br>(ms) | FP16<br>Batch Size=8<br>(ms) | FP32<br>Batch Size=1<br>(ms) | FP32<br>Batch Size=4<br>(ms) | FP32<br>Batch Size=8<br>(ms) |
|---------------------------|-----------|-------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| ResNeXt101_<br>32x8d_wsl      | 224       | 256               | 18.19374                     | 21.93529                     | 34.67802                     | 18.52528                     | 34.25319                     | 67.2283                      |
| ResNeXt101_<br>32x16d_wsl     | 224       | 256               | 18.52609                     | 36.8288                      | 62.79947                     | 25.60395                     | 71.88384                     | 137.62327                    |
| ResNeXt101_<br>32x32d_wsl     | 224       | 256               | 33.51391                     | 70.09682                     | 125.81884                    | 54.87396                     | 160.04337                    | 316.17718                    |
| ResNeXt101_<br>32x48d_wsl     | 224       | 256               | 50.97681                     | 137.60926                    | 190.82628                    | 99.01698256                  | 315.91261                    | 551.83695                    |
| Fix_ResNeXt101_<br>32x48d_wsl | 320       | 320               | 78.62869                     | 191.76039                    | 317.15436                    | 160.0838242                  | 595.99296                    | 1151.47384                   |
| EfficientNetB0            | 224       | 256               | 3.40122                      | 5.95851                      | 9.10801                      | 3.442                        | 6.11476                      | 9.3304                       |
| EfficientNetB1            | 240       | 272               | 5.25172                      | 9.10233                      | 14.11319                     | 5.3322                       | 9.41795                      | 14.60388                     |
| EfficientNetB2            | 260       | 292               | 5.91052                      | 10.5898                      | 17.38106                     | 6.29351                      | 10.95702                     | 17.75308                     |
| EfficientNetB3            | 300       | 332               | 7.69582                      | 16.02548                     | 27.4447                      | 7.67749                      | 16.53288                     | 28.5939                      |
| EfficientNetB4            | 380       | 412               | 11.55585                     | 29.44261                     | 53.97363                     | 12.15894                     | 30.94567                     | 57.38511                     |
| EfficientNetB5            | 456       | 488               | 19.63083                     | 56.52299                     | -                            | 20.48571                     | 61.60252                     | -                            |
| EfficientNetB6            | 528       | 560               | 30.05911                     | -                            | -                            | 32.62402                     | -                            | -                            |
| EfficientNetB7            | 600       | 632               | 47.86087                     | -                            | -                            | 53.93823                     | -                            | -                            |
| EfficientNetB0_small      | 224       | 256               | 2.39166                      | 4.36748                      | 6.96002                      | 2.3076                       | 4.71886                      | 7.21888                      |

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/EfficientNet/`、 `ppcls/configs/ImageNet/ResNeXt/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

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

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../inference_deployment/cpp_deploy.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../inference_deployment/cpp_deploy_on_windows.md)完成相应的预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../inference_deployment/paddle_serving_deploy.md)来完成相应的部署工作。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../inference_deployment/paddle_lite_deploy.md)来完成相应的部署工作。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](@shuilong)来完成相应的部署工作。
