# YOLO26 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
- [2. 模型快速体验](#2)
- [3. 模型训练、评估和预测](#3)
- [4. 预处理与权重转换说明](#4)
- [5. 模型推理部署](#5)

<a name='1'></a>

## 1. 模型介绍

<a name='1.1'></a>

### 1.1 模型简介

[YOLO26](https://docs.ultralytics.com/models/yolo26/) 是 Ultralytics YOLO 系列模型之一，支持检测、实例分割、姿态估计、旋转框检测和图像分类等任务。检测模型侧重端到端 NMS-Free 推理、去除 DFL、双头结构和面向边缘设备的部署效率；分类模型则使用 `-cls` 后缀，例如 `yolo26n-cls.pt`。

PaddleClas 当前实现的是 YOLO26 的图像分类版本，结构与 Ultralytics `yolo26-cls.yaml` 对齐，主要由 `ConvBNAct`、`C3k2`、`C2PSA` 和 `ClassifyHead` 组成。分类头采用 `conv -> pool -> dropout -> linear`，其中 `linear` 参数名与 Ultralytics 权重中的 `model.10.linear.*` 保持一致，便于直接转换并加载官方 PyTorch 权重。

<a name='1.2'></a>

### 1.2 模型指标

下表中 `Reference` 指标来自 Ultralytics 官方 ImageNet 分类结果；`PaddleClas` 指标为本仓库按官方预处理复现的结果。当前已完整复现 `YOLO26n-cls`，其它 scale 可使用同一转换脚本生成 Paddle 权重后评估。

| Models | Top1 | Top5 | Reference<br>Top1 | Reference<br>Top5 | FLOPs<br>(B) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YOLO26n-cls | 0.714 | 0.901 | 0.714 | 0.901 | 0.5 | 2.8 |
| YOLO26s-cls | 0.759 | 0.929 | 0.760 | 0.929 | 1.6 | 6.7 |
| YOLO26m-cls | 0.780 | 0.942 | 0.781 | 0.942 | 4.9 | 11.6 |
| YOLO26l-cls | 0.791 | 0.946 | 0.790 | 0.946 | 6.2 | 14.1 |
| YOLO26x-cls | 0.799 | 0.950 | 0.799 | 0.950 | 13.6 | 29.6 |

**备注：**
1. YOLO26 官方分类模型使用 ImageNet 1k 验证集评估，输入尺寸为 224。
2. FLOPs 与 Params 为 Ultralytics 官方给出的 fused model 指标。
3. PaddleClas 复现指标依赖与官方一致的验证预处理：`Resize(224, bilinear, PIL) -> CenterCrop(224) -> scale 1/255 -> mean 0/std 1`。

<a name="2"></a>

## 2. 模型快速体验

安装 PaddlePaddle 和 PaddleClas 后，可以使用 `tools/infer.py` 对图片进行预测。体验方法可以参考 [ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

YOLO26 的配置文件位于：

```bash
ppcls/configs/ImageNet/YOLO26/
```

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet 数据准备、训练、评估、预测等内容。通用流程可以参考 [ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

使用转换后的 `YOLO26n-cls` 权重评估 ImageNet 验证集的示例命令如下：

```bash
python tools/eval.py \
  -c ppcls/configs/ImageNet/YOLO26/YOLO26n.yaml \
  -o Global.pretrained_model=/path/to/yolo26n-cls.pdparams \
  -o DataLoader.Eval.dataset.image_root=/path/to/imagenet \
  -o DataLoader.Eval.dataset.cls_label_path=/path/to/imagenet/val_list.txt
```

<a name="4"></a>

## 4. 预处理与权重转换说明

YOLO26 官方分类权重自带的验证与推理预处理为：

```text
Resize(size=224, interpolation=bilinear, antialias=True)
CenterCrop(size=(224, 224))
ToTensor()
Normalize(mean=[0, 0, 0], std=[1, 1, 1])
```

因此 PaddleClas 配置中需要使用如下 Eval/Infer 预处理：

```yaml
- ResizeImage:
    resize_short: 224
    interpolation: bilinear
    backend: pil
- CropImage:
    size: 224
- NormalizeImage:
    scale: 1.0/255.0
    mean: [0.0, 0.0, 0.0]
    std: [1.0, 1.0, 1.0]
    order: ''
```

如果使用 `cv2` resize 或常规 ImageNet mean/std，会造成输入分布与官方权重不一致，影响验证精度。

从 Ultralytics `.pt` 转换为 Paddle `.pdparams` 时，需要保留 `model.0.*` 到 `model.10.linear.*` 的参数命名，并完成以下映射：

```text
running_mean -> _mean
running_var  -> _variance
Linear weight: [out, in] -> [in, out]
```


<a name="5"></a>

## 5. 模型推理部署

Paddle Inference 是飞桨的原生推理库，作用于服务器端和云端，提供高性能推理能力。相比于直接基于预训练模型进行预测，Paddle Inference 可使用 MKLDNN、CUDNN、TensorRT 进行预测加速。

YOLO26 分类模型的推理部署流程与其它图像分类模型一致，可以参考：

- [推理模型准备](./ResNet.md#41-推理模型准备)
- [基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理)
- [模型服务化部署](../../deployment/image_classification/paddle_serving.md)
- [端侧部署](../../deployment/image_classification/paddle_lite.md)
- [Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)
