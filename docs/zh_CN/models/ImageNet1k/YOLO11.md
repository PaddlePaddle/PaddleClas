# YOLO11 分类模型
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
- [2. 模型快速体验](#2)
- [3. 模型训练、评估和预测](#3)
    - [3.1 配置文件](#3.1)
    - [3.2 评估](#3.2)
    - [3.3 预测](#3.3)
    - [3.4 官方权重转换与前向对齐](#3.4)
    - [3.5 前向对齐结果](#3.5)
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

YOLO11 是 Ultralytics 发布的 YOLO11 系列模型，其中 `YOLO11-cls` 为 ImageNet 1K 图像分类模型，输入分辨率为 `224x224`。该系列分类模型沿用了 YOLO11 的基础卷积设计，并在分类任务中使用 `C3k2`、`C2PSA` 和 `Classify` Head 等模块，在精度和推理效率之间取得了较好的平衡。

PaddleClas 当前提供 `YOLO11_cls_n/s/m/l/x` 五个分类规格，均由 Ultralytics 官方 `yolo11*-cls.pt` 权重转换得到。

<a name='1.2'></a>

### 1.2 模型指标

使用 Ultralytics 官方 `yolo11*-cls.pt` 权重和相同输入进行前向对齐，比较 Paddle 模型与源模型的末级特征 `feat` 和分类输出 `out`，结果如下：

| Models | feat<br>max_abs_diff | out<br>max_abs_diff |
|:--:|:--:|:--:|
| YOLO11_cls_n | 4.23e-6 | 3.81e-6 |
| YOLO11_cls_s | 9.48e-6 | 9.66e-6 |
| YOLO11_cls_m | 3.24e-6 | 5.96e-6 |
| YOLO11_cls_l | 6.14e-6 | 3.34e-6 |
| YOLO11_cls_x | 3.70e-6 | 3.10e-6 |

最大绝对误差稳定在 `1e-5` 以内。

在ImageNet数据集下的精度测试结果如下：

| Models | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YOLO11_cls_n | 0.7005 | 0.8939 | 0.700 | 0.894 | 0.5 | 1.63 |
| YOLO11_cls_s | 0.7540 | 0.9264 | 0.754 | 0.927 | 1.6 | 5.55 |
| YOLO11_cls_m | 0.7740 | 0.9384 | 0.773 | 0.939 | 5.0 | 10.46 |
| YOLO11_cls_l | 0.7834 | 0.9422 | 0.783 | 0.943 | 6.2 | 12.94 |
| YOLO11_cls_x | 0.7943 | 0.9488 | 0.795 | 0.949 | 13.7 | 28.46 |

**说明：**

- `Reference top1/top5` 来自 Ultralytics 官方 YOLO11 文档与官方 `pt` 权重实测结果。
- `Top1/Top5` 为 PaddleClas 使用转换后的 Paddle 权重在 ImageNet1k `val` 集上实测结果。
- 由于 Ultralytics 的分类评估预处理为 `Resize(224) -> CenterCrop(224) -> ToTensor()`，不使用 ImageNet mean/std 标准化，因此 PaddleClas 配置已按该预处理对齐。

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

<a name="3.1"></a>

### 3.1 配置文件

YOLO11 分类模型配置位于 `ppcls/configs/ImageNet/YOLO11/` 目录下：

- `YOLO11_cls_n.yaml`
- `YOLO11_cls_s.yaml`
- `YOLO11_cls_m.yaml`
- `YOLO11_cls_l.yaml`
- `YOLO11_cls_x.yaml`

启动训练、评估和预测的方法可参考[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

<a name="3.2"></a>

### 3.2 评估

以 `YOLO11_cls_n` 为例，可使用如下命令在 ImageNet1k 验证集上评估：

```bash
python tools/eval.py \
  -c ppcls/configs/ImageNet/YOLO11/YOLO11_cls_n.yaml \
  -o Global.pretrained_model=/path/to/YOLO11_cls_n_pretrained.pdparams
```

其余规格仅需替换配置文件和权重路径即可。

<a name="3.3"></a>

### 3.3 预测

以 `YOLO11_cls_s` 为例，可使用如下命令进行单张图片预测：

```bash
python tools/infer.py \
  -c ppcls/configs/ImageNet/YOLO11/YOLO11_cls_s.yaml \
  -o Global.pretrained_model=/path/to/YOLO11_cls_s_pretrained.pdparams \
  -o Infer.infer_imgs=docs/images/inference_deployment/whl_demo.jpg
```

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库，作用于服务器端和云端，提供高性能推理能力。相比于直接基于预训练模型进行预测，Paddle Inference 可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于 Paddle Inference 的介绍，可以参考[Paddle Inference 官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备)。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 Python 预测引擎推理的示例。您可以参考[ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理)。

<a name="4.3"></a>

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md)完成相应的预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于 Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../../deployment/image_classification/paddle_serving.md)来完成相应的部署工作。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../../deployment/image_classification/paddle_lite.md)来完成相应的部署工作。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转换到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括 TensorRT、OpenVINO、MNN、TNN、NCNN 以及其他支持 ONNX 开源格式的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换和推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)来完成相应工作。
