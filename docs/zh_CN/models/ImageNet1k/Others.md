# 其他模型
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

2012 年，Alex 等人提出的 AlexNet 网络在 ImageNet 大赛上以远超第二名的成绩夺冠，卷积神经网络乃至深度学习引起了广泛的关注。AlexNet 使用 relu 作为 CNN 的激活函数，解决了 sigmoid 在网络较深时的梯度弥散问题。训练时使用 Dropout 随机丢掉一部分神经元，避免了模型过拟合。网络中使用重叠的最大池化代替了此前 CNN 中普遍使用的平均池化，避免了平均池化的模糊效果，提升了特征的丰富性。从某种意义上说，AlexNet 引爆了神经网络的研究与应用热潮。

SqueezeNet 在 ImageNet-1k 上实现了与 AlexNet 相同的精度，但只用了 1/50 的参数量。该网络的核心是 Fire 模块，Fire 模块通过使用 1x1 的卷积实现通道降维，从而大大节省了参数量。作者通过大量堆叠 Fire 模块组成了 SqueezeNet。

VGG 由牛津大学计算机视觉组和 DeepMind 公司研究员一起研发的卷积神经网络。该网络探索了卷积神经网络的深度和其性能之间的关系，通过反复的堆叠 3x3 的小型卷积核和 2x2 的最大池化层，成功的构建了多层卷积神经网络并取得了不错的收敛精度。最终，VGG 获得了 ILSVRC 2014 比赛分类项目的亚军和定位项目的冠军。

DarkNet53 是 YOLO 作者在论文设计的用于目标检测的 backbone，该网络基本由 1x1 与 3x3 卷积构成，共 53 层，取名为 DarkNet53。

<a name='1.2'></a>

### 1.2 模型指标

| Models                    | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| AlexNet                   | 0.567  | 0.792  | 0.5720            |                   | 1.370        | 61.090            |
| SqueezeNet1_0             | 0.596  | 0.817  | 0.575             |                   | 1.550        | 1.240             |
| SqueezeNet1_1             | 0.601  | 0.819  |                   |                   | 0.690        | 1.230             |
| VGG11                     | 0.693  | 0.891  |                   |                   | 15.090       | 132.850           |
| VGG13                     | 0.700  | 0.894  |                   |                   | 22.480       | 133.030           |
| VGG16                     | 0.720  | 0.907  | 0.715             | 0.901             | 30.810       | 138.340           |
| VGG19                     | 0.726  | 0.909  |                   |                   | 39.130       | 143.650           |
| DarkNet53                 | 0.780  | 0.941  | 0.772             | 0.938             | 18.580       | 41.600            |

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models                 | Crop Size | Resize Short Size | FP32<br/>Batch Size=1<br/>(ms) | FP32<br/>Batch Size=4<br/>(ms) | FP32<br/>Batch Size=8<br/>(ms) |
|---------------------------|-----------|-------------------|-------------------|-------------------|-------------------|
| AlexNet                   | 224       | 256               | 0.81           | 1.50           | 2.33           |
| SqueezeNet1_0             | 224       | 256               | 0.68           | 1.64           | 2.62           |
| SqueezeNet1_1             | 224       | 256               | 0.62           | 1.30           | 2.09           |
| VGG11                     | 224       | 256               | 1.72           | 4.15           | 7.24           |
| VGG13                     | 224       | 256               | 2.02           | 5.28           | 9.54           |
| VGG16                     | 224       | 256               | 2.48           | 6.79           | 12.33          |
| VGG19                     | 224       | 256               | 2.93           | 8.28           | 15.21          |
| DarkNet53                 | 256       | 256               | 2.79           | 6.42           | 10.89          |

<a name='1.3.2'></a>

#### 1.3.2 基于 T4 GPU 的预测速度

| Models                | Crop Size | Resize Short Size | FP16<br>Batch Size=1<br>(ms) | FP16<br>Batch Size=4<br>(ms) | FP16<br>Batch Size=8<br>(ms) | FP32<br>Batch Size=1<br>(ms) | FP32<br>Batch Size=4<br>(ms) | FP32<br>Batch Size=8<br>(ms) |
|-----------------------|-----------|-------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| AlexNet               | 224       | 256               | 1.06447                      | 1.70435                      | 2.38402                      | 1.44993                      | 2.46696                      | 3.72085                      |
| SqueezeNet1_0         | 224       | 256               | 0.97162                      | 2.06719                      | 3.67499                      | 0.96736                      | 2.53221                      | 4.54047                      |
| SqueezeNet1_1         | 224       | 256               | 0.81378                      | 1.62919                      | 2.68044                      | 0.76032                      | 1.877                        | 3.15298                      |
| VGG11                 | 224       | 256               | 2.24408                      | 4.67794                      | 7.6568                       | 3.90412                      | 9.51147                      | 17.14168                     |
| VGG13                 | 224       | 256               | 2.58589                      | 5.82708                      | 10.03591                     | 4.64684                      | 12.61558                     | 23.70015                     |
| VGG16                 | 224       | 256               | 3.13237                      | 7.19257                      | 12.50913                     | 5.61769                      | 16.40064                     | 32.03939                     |
| VGG19                 | 224       | 256               | 3.69987                      | 8.59168                      | 15.07866                     | 6.65221                      | 20.4334                      | 41.55902                     |
| DarkNet53             | 256       | 256               | 3.18101                      | 5.88419                      | 10.14964                     | 4.10829                      | 12.1714                      | 22.15266                     |

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、SwinTransformer 在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/SwinTransformer/` 中提供了 SwinTransformer 的训练配置，可以通过如下脚本启动训练：此部分内容可以参考[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

**备注：** 由于 SwinTransformer 系列模型默认使用的 GPU 数量为 8 个，所以在训练时，需要指定8个GPU，如`python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c xxx.yaml`, 如果使用 4 个 GPU 训练，默认学习率需要减小一半，精度可能有损。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备) 。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 python 预测引擎推理的示例。您可以参考[ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理) 对 SwinTransformer 完成推理预测。

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
