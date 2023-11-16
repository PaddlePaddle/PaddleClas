# PP-HGNeV2 系列
---
- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型细节](#1.2)
    - [1.3 实验精度](#1.3)
- [2. 模型训练、评估和预测](#2)
    - [2.1 环境配置](#2.1)
    - [2.2 数据准备](#2.2)
    - [2.3 模型训练](#2.3)
      - [2.3.1 从头训练 ImageNet](#2.3.1)
      - [2.3.2 基于 ImageNet 权重微调其他分类任务](#2.3.2)
    - [2.4 模型评估](#2.4)
    - [2.5 模型预测](#2.5)
- [3. 模型推理部署](#3)
  - [3.1 推理模型准备](#3.1)
  - [3.2 基于 Python 预测引擎推理](#3.2)
    - [3.2.1 预测单张图像](#3.2.1)
    - [3.2.2 基于文件夹的批量预测](#3.2.2)
  - [3.3 基于 C++ 预测引擎推理](#3.3)
  - [3.4 服务化部署](#3.4)
  - [3.5 端侧部署](#3.5)
  - [3.6 Paddle2ONNX 模型转换与预测](#3.6)

<a name='1'></a>

## 1. 模型介绍

<a name='1.1'></a>

### 1.1 模型简介

PP-HGNetV2(High Performance GPU Network V2) 是百度飞桨视觉团队自研的 PP-HGNet 的下一代版本，其在 PP-HGNet 的基础上，做了进一步优化和改进，最终在 NVIDIA GPU 设备上，将 "Accuracy-Latency Balance" 做到了极致，精度大幅超过了其他同样推理速度的模型。其在单标签分类、多标签分类、目标检测、语义分割等任务中，均有较强的表现，与常见的服务器端模型在精度-预测耗时的比较如下图所示。

![](../../../images/models/V100_benchmark/v100.fp32.bs1.main_fps_top1_s.png)

* GPU 评估环境基于 V100 机器，在 FP32+TensorRT 配置下运行 2100 次测得（去除前 100 次的 warmup 时间）。

<a name='1.2'></a>

### 1.2 模型细节

PP-HGNetV2 在 PP-HGNet 上的具体改进点如下：

- 改进了 PPHGNet 网络 stem 部分，堆叠更多的 2x2 卷积核以学习更丰富的局部特征，使用更小的通道数以提升大分辨率任务如目标检测、语义分割等的推理速度；
- 替换了 PP-HGNet 中靠后 stage 的较冗余的标准卷积层为 PW + DW5x5 组合，在获得更大感受野的同时网络的参数量更少，且精度可以进一步提升；
- 增加了 LearnableAffineBlock 模块，其可以在增加极少参数量的同时大幅提升较小模型的精度，且对推理时间无损；
- 重构了 PP-HGNet 网络的 stage 分布，使其涵盖了从 B0-B6 不同量级的模型，从而满足不同任务的需求。

除以上改进点之外，相比 PaddleClas 提供的其他模型，PP-HGNetV2 默认提供了精度更高、泛化能力更强的 [SSLD](https://arxiv.org/abs/2103.05959) 预训练权重，其在下游任务中表现更佳。

<a name='1.3'></a>

### 1.3 模型精度

PP-HGNetV2 的精度、速度指标、预训练权重、推理模型权重链接如下：

| Model | Top-1 Acc(\%)(stage-2) | Top-5 Acc(\%)(stage-2) | Latency(ms) | stage-1预训练模型下载地址 | stage-2预训练模型下载地址 |inference模型下载地址(stage-2) |
|:--: |:--: |:--: |:--: | :--: |:--: |:--: |
| PPHGNetV2_B0     | 77.77 | 93.91 | 0.52 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B0_ssld_stage1_pretrained.pdparams)| [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B0_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B0_ssld_infer.tar) |
| PPHGNetV2_B1     | 79.18 | 94.57 | 0.58 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B1_ssld_stage1_pretrained.pdparams)| [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B1_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B1_ssld_infer.tar) |
| PPHGNetV2_B2     | 81.74 | 95.88 | 0.95 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B2_ssld_stage1_pretrained.pdparams)| [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B2_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B2_ssld_infer.tar) |
| PPHGNetV2_B3     | 82.98 | 96.43 | 1.18 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B3_ssld_stage1_pretrained.pdparams)| [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B3_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B3_ssld_infer.tar) |
| PPHGNetV2_B4     | 83.57 | 96.72 | 1.46 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B4_ssld_stage1_pretrained.pdparams)| [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B4_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B4_ssld_infer.tar) |
| PPHGNetV2_B5     | 84.75 | 97.32 | 2.84 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B5_ssld_stage1_pretrained.pdparams)| [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B5_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B5_ssld_infer.tar) |
| PPHGNetV2_B6     | 86.30 | 97.84 | 5.29 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B6_ssld_stage1_pretrained.pdparams)| [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B6_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B6_ssld_infer.tar) |

**备注：**

* 测试环境：V100，FP32+TensorRT8.5，BS=1；
* 为了让下游任务有更高的精度，PP-HGNetV2 全系列提供了 `SSLD` 预训练权重。关于 `SSLD` 相关的内容介绍和训练方法，可以查看[SSLD paper](https://arxiv.org/abs/2103.05959)、[SSLD 训练](../../training/advanced/knowledge_distillation.md)，此处提供的 stage-1 的权重为 `SSLD` 的 stage-1 阶段使用 ImageNet1k+ImageNet22k 挖掘数据蒸馏训练得到的权重，stage-2 权重为 `SSLD` 的 stage-2 阶段使用 ImageNet1k 蒸馏微调得到的权重。在实际场景中，stage-1 的权重有更好的泛化性，建议直接使用 stage-1 的权重来做下游任务训练。


<a name="2"></a>

## 2. 模型训练、评估和预测

<a name="2.1"></a>  

### 2.1 环境配置

* 安装：请先参考文档[环境准备](../../installation.md) 配置 PaddleClas 运行环境。

<a name="2.2"></a>

### 2.2 数据准备

请在[ImageNet 官网](https://www.image-net.org/)准备 ImageNet-1k 相关的数据。


进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，将下载好的数据命名为 `ILSVRC2012` ，存放于此。 `ILSVRC2012` 目录中具有以下数据：

```
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
├── train_list.txt
...
├── val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
├── val_list.txt
```

其中 `train/` 和 `val/` 分别为训练集和验证集。`train_list.txt` 和 `val_list.txt` 分别为训练集和验证集的标签文件。

**备注：**

* 关于 `train_list.txt`、`val_list.txt`的格式说明，可以参考[PaddleClas分类数据集格式说明](../../training/single_label_classification/dataset.md#1-数据集格式说明) 。


<a name="2.3"></a>

### 2.3 模型训练

<a name="2.3.1"></a>

#### 2.3.1 从头训练 ImageNet

在 `ppcls/configs/ImageNet/PPHGNetV2/` 中提供了 PPHGNetV2 不同大小模型的训练配置，可以加载对应模型的配置训练。如训练 `PPHGNetV2_B4`，则可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/train.py \
        -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
        -o Global.output_dir=./output/PPHGNetV2_B4 \
        -o Arch.pretrained=False
```


**备注：**

* 当前精度最佳的模型会保存在 `output/PPHGNetV2_B4/best_model.pdparams`;
* 此处只是展示了如何从头训练 ImageNet数据，该配置并未使用激进的训练策略或者蒸馏训练策略，所以训练得到的精度较 [1.3](#1.3) 小节要低。如果希望得到 [1.3](#1.3) 小节中的精度，可以查看[SSLD 训练](../../training/advanced/knowledge_distillation.md)，配置好相关的数据，加载 [stage-1 配置](../../../../ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4_ssld_stage1.yaml)、[stage-2 配置](../../../../PPHGNetV2_B4_ssld_stage2.yaml)训练即可。

<a name="2.3.2"></a>

#### 2.3.2 基于 ImageNet 权重微调其他分类任务

模型微调时，需要加载预训练权重，同时需要缩小学习率，以免破坏原有权重。如微调训练 `PPHGNetV2_B4`，则可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/train.py \
        -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
        -o Global.epochs=30 \
        -o Global.output_dir=./output/PPHGNetV2_B4 \
        -o Optimizer.lr.learning_rate=0.05

```
**备注：**

* `epochs` 和 `learning_rate` 可以根据实际情况调整；
* 为了更好的泛化性，此处默认加载的权重为 `SSLD` stage-1 训练得到的权重。

<a name="2.4"></a>

### 2.4 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```shell
python tools/eval.py \
    -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
    -o Global.pretrained_model=output/PPHGNetV2_B4/best_model
```

其中 `-o Global.pretrained_model="output/PPHGNetV2_B4/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

<a name="2.5"></a>

### 2.5 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```shell
python tools/infer.py \
    -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
    -o Global.pretrained_model=output/PPHGNetV2_B4/best_model
```

输出结果如下：

```
 [{'class_ids': [8, 7, 86, 82, 83], 'scores': [0.92473, 0.07478, 0.00025, 7e-05, 6e-05], 'file_name': 'docs/images/inference_deployment/whl_demo.jpg', 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'prairie chicken, prairie grouse, prairie fowl']}]
```

**备注：**

* 这里`-o Global.pretrained_model="output/PPHGNetV2_B4/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

* 默认是对 `docs/images/inference_deployment/whl_demo.jpg` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。

* 默认输出的是 Top-5 的值，如果希望输出 Top-k 的值，可以指定`-o Infer.PostProcess.topk=k`，其中，`k` 为您指定的值。

* 默认的标签映射基于 ImageNet 数据集，如果改变数据集，需要重新指定`Infer.PostProcess.class_id_map_file`，该映射文件的制作方法可以参考`ppcls/utils/imagenet1k_label_list.txt`。



<a name="3"></a>

## 3. 模型推理部署

<a name="3.1"></a>

### 3.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。


此处，我们提供了将权重和模型转换的脚本，执行该脚本可以得到对应的 inference 模型：

```shell
python3 tools/export_model.py \
    -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
    -o Global.pretrained_model=output/PPHGNetV2_B4/best_model \
    -o Global.save_inference_dir=deploy/models/PPHGNetV2_B4_infer
```
执行完该脚本后会在 `deploy/models/` 下生成 `PPHGNetV2_B4_infer` 文件夹，`models` 文件夹下应有如下文件结构：

```
├── PPHGNetV2_B4_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="3.2"></a>

### 3.2 基于 Python 预测引擎推理


<a name="3.2.1"></a>  

#### 3.2.1 预测单张图像

返回 `deploy` 目录：

```
cd ../
```

运行下面的命令，对图像 `./images/ImageNet/ILSVRC2012_val_00000010.jpeg` 进行分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPHGNetV2_B4_infer
# 使用下面的命令使用 CPU 进行预测
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPHGNetV2_B4_infer -o Global.use_gpu=False
```

输出结果如下。

```
ILSVRC2012_val_00000010.jpeg:    class id(s): [332, 153, 283, 338, 265], score(s): [0.94, 0.03, 0.02, 0.00, 0.00], label_name(s): ['Angora, Angora rabbit', 'Maltese dog, Maltese terrier, Maltese', 'Persian cat', 'guinea pig, Cavia cobaya', 'toy poodle']
```

<a name="3.2.2"></a>  

#### 3.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPHGNetV2_B4_infer -o Global.infer_imgs=images/ImageNet/
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
ILSVRC2012_val_00000010.jpeg:    class id(s): [332, 153, 283, 338, 265], score(s): [0.94, 0.03, 0.02, 0.00, 0.00], label_name(s): ['Angora, Angora rabbit', 'Maltese dog, Maltese terrier, Maltese', 'Persian cat', 'guinea pig, Cavia cobaya', 'toy poodle']
ILSVRC2012_val_00010010.jpeg:    class id(s): [626, 487, 531, 622, 593], score(s): [0.81, 0.08, 0.03, 0.01, 0.01], label_name(s): ['lighter, light, igniter, ignitor', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'digital watch', 'lens cap, lens cover', 'harmonica, mouth organ, harp, mouth harp']
ILSVRC2012_val_00020010.jpeg:    class id(s): [178, 211, 246, 236, 181], score(s): [1.00, 0.00, 0.00, 0.00, 0.00], label_name(s): ['Weimaraner', 'vizsla, Hungarian pointer', 'Great Dane', 'Doberman, Doberman pinscher', 'Bedlington terrier']
ILSVRC2012_val_00030010.jpeg:    class id(s): [80, 83, 23, 8, 81], score(s): [1.00, 0.00, 0.00, 0.00, 0.00], label_name(s): ['black grouse', 'prairie chicken, prairie grouse, prairie fowl', 'vulture', 'hen', 'ptarmigan']
```

<a name="3.3"></a>

### 3.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md)完成相应的预测库编译和模型预测工作。

<a name="3.4"></a>

### 3.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../../deployment/image_classification/paddle_serving.md)来完成相应的部署工作。

<a name="3.5"></a>

### 3.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../../deployment/image_classification/paddle_lite.md)来完成相应的部署工作。

<a name="3.6"></a>

### 3.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)来完成相应的部署工作。
