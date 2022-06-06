# PULC语种分类模型

## 目录

- [1.  模型和应用场景介绍](#1)
- [2.  模型快速体验](#2)
- [3.  模型训练、评估和预测](#3)
  - [3.1 环境配置](#3.1)  
  - [3.2 数据准备](#3.2)
    - [3.2.1 数据集来源](#3.2.1)
    - [3.2.2 数据集获取](#3.2.2)
  - [3.3 模型训练](#3.3)
  - [3.4 模型评估](#3.4)
  - [3.5 模型预测](#3.5)
- [4. 模型压缩](#4)
  - [4.1 SKL-UGI 知识蒸馏](#4.1)
    - [4.1.1 教师模型训练](#4.1.1)
    - [4.1.2 蒸馏训练](#4.1.2)
- [5. 超参搜索](#5)
- [6. 模型推理部署](#6)
  - [6.1 推理模型准备](#6.1)
    - [6.1.1 基于训练得到的权重导出 inference 模型](#6.1.1)
    - [6.1.2 直接下载 inference 模型](#6.1.2)
  - [6.2 基于 Python 预测引擎推理](#6.2)
    - [6.2.1 预测单张图片](#6.2.1)
    - [6.2.2 基于文件夹的批量预测](#6.2.2)
  - [6.3 基于 C++ 预测引擎推理](#6.3)
  - [6.4 服务化部署](#6.4)
  - [6.5 端侧部署](#6.5)
  - [6.6 Paddle2ONNX 模型转换与预测](#6.6)

<a name="1"></a>

## 1. 模型和应用场景介绍

该案例提供了用户使用 PaddleClas 的超轻量图像分类方案（PULC，Practical Ultra Lightweight Classification）快速构建轻量级、高精度、可落地的语种分类模型。使用该方法训练得到的模型可以快速判断图片中的文字语种，该模型可以广泛应用于金融、政务等各种涉及多语种OCR处理的场景中。

下表列出了语种分类模型的相关指标，前两行展现了使用 SwinTranformer_tiny 和 MobileNetV3_large_x1_0 作为 backbone 训练得到的模型的相关指标，第三行至第六行依次展现了替换 backbone 为 PPLCNet_x1_0、使用 SSLD 预训练模型、使用 SSLD 预训练模型 + EDA 策略、使用 SSLD 预训练模型 + EDA 策略 + SKL-UGI 知识蒸馏策略训练得到的模型的相关指标。其中替换 backbone 为 PPLCNet_x1_0时，将数据预处理时的输入尺寸变为[192,48]，且网络的下采样stride调整为[2, [2, 1], [2, 1], [2, 1], [2, 1]]。

| 模型                   | 精度      | 延时     | 存储    | 策略                                           |
| ---------------------- | --------- | -------- | ------- | ---------------------------------------------- |
| SwinTranformer_tiny    | 98.12     | 166.64   | 107     | 使用ImageNet预训练模型                         |
| MobileNetV3_large_x1_0 | 98.3      | 4.78     | 17      | 使用ImageNet预训练模型                         |
| PPLCNet_x1_0           | 98.35     | 2.56     | 6.5     | 使用ImageNet预训练模型                         |
| PPLCNet_x1_0           | 98.7      | 2.56     | 6.5     | 使用SSLD预训练模型                             |
| PPLCNet_x1_0           | 99.12     | 2.56     | 6.5     | 使用SSLD预训练模型+EDA策略                     |
| **PPLCNet_x1_0**       | **99.26** | **2.56** | **6.5** | 使用SSLD预训练模型+EDA策略+SKL-UGI知识蒸馏策略 |

从表中可以看出，backbone 为 SwinTranformer_tiny 时精度比较高，但是推理速度较慢。将 backboone 替换为轻量级模型 MobileNetV3_large_x1_0 后，精度和速度都有了提升。将 backbone 替换为 PPLCNet_x1_0 且调整预处理输入尺寸和网络的下采样stride时，精度较 MobileNetV3_large_x1_0 高0.05个百分点，同时速度提升 2 倍左右。在此基础上，使用 SSLD 预训练模型后，在不改变推理速度的前提下，精度可以提升 0.35 个百分点，进一步地，当融合EDA策略后，精度可以再提升 0.42 个百分点，最后，在使用 SKL-UGI 知识蒸馏后，精度可以继续提升 0.14 个百分点。此时，PPLCNet_x1_0 超过了 MobileNetV3_large_x1_0 和 SwinTranformer_tiny 模型的精度，并且速度有了明显提升。关于 PULC 的训练方法和推理部署方法将在下面详细介绍。

**备注：**关于PPLCNet的介绍可以参考[PPLCNet介绍](../models/PP-LCNet.md)，相关论文可以查阅[PPLCNet paper](https://arxiv.org/abs/2109.15099)。

<a name="2"></a>

## 2. 模型快速体验

​    （pip方式，待补充）

<a name="3"></a>

## 3. 模型训练、评估和预测

<a name="3.1"></a>

### 3.1 环境配置

- 安装：请先参考 [Paddle 安装教程](../installation/install_paddle.md) 以及 [PaddleClas 安装教程](../installation/install_paddleclas.md) 配置 PaddleClas 运行环境。

<a name="3.2"></a>

### 3.2 数据准备

<a name="3.2.1"></a>

#### 3.2.1 数据集来源

[第1节](#1)中提供的模型使用内部数据训练得到，该数据集暂时不方便公开。这里基于 [Multi-lingual scene text detection and recognition](https://rrc.cvc.uab.es/?ch=15&com=downloads) 开源数据集构造了一个多语种demo数据集，用于体验本案例的预测过程。

![](../../images/PULC/docs/language_classification_original_data.png)

<a name="3.2.2"></a>

#### 3.2.2 数据集获取

[第1节](#1)中提供的模型共支持10个类别，分别为：

`0` 表示阿拉伯语（arabic）；`1` 表示中文繁体（chinese_cht）；`2` 表示斯拉夫语（cyrillic）；`3` 表示梵文（devanagari）；`4` 表示日语（japan）；`5` 表示卡纳达文（ka）；`6` 表示韩语（korean）；`7` 表示泰米尔文（ta）；`8` 表示泰卢固文（te）；`9` 表示拉丁语（latin）。

在 Multi-lingual scene text detection and recognition 数据集中，仅包含了阿拉伯语、日语、韩语和拉丁语数据，这里分别将4个语种的数据各抽取100张作为本案例的demo数据，并赋予对应的标签。

如果想要制作自己的多语种数据集，可以按照需求收集并整理自己任务中需要语种的数据，此处提供了经过上述方法处理好的demo数据，可以直接下载得到。

**备注：**语种分类任务中的图片数据需要将整图中的文字区域抠取出来，仅仅使用文本行部分作为图片数据。

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，下载并解压多语种场景的demo数据。

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/cls_demo/language_classification_demo.tar
tar -xf language_classification_demo.tar
cd ../
```

执行上述命令后，`dataset/`下存在`language_classification_demo`目录，该目录中具有以下数据：

```
├── img
│   ├── word_1.png
│   ├── word_2.png
...
├── label.txt
└── label_list.txt
```

其中`img/`存放了4种语种的400张测试数据。`label.txt`为图片对应的的标签文件，`label_list.txt`是10类语种分类模型对应的类别列表，用这些图片可以快速体验本案例中模型的预测过程。

***备注：***

-  这里的`label_list.txt`是10类语种分类模型对应的类别列表，如果自己构造的数据集语种类别发生变化，需要自行调整。
-  如果想要自己构造训练集和验证集，可以参考[PaddleClas分类数据集格式说明](../data_preparation/classification_dataset.md#1-数据集格式说明) 。

<a name="3.3"></a>

### 3.3 模型训练

在`ppcls/configs/PULC/language_classification/PPLCNet_x1_0.yaml`中提供了基于该场景的训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/language_classification/PPLCNet_x1_0.yaml
```

<a name="3.4"></a>

### 3.4 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/language_classification/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

其中 `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

<a name="3.5"></a>

### 3.5 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```bash
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/language_classification/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

输出结果如下：

```
[{'class_ids': [4, 9], 'scores': [0.96809, 0.01001], 'file_name': 'deploy/images/PULC/language_classification/word_35404.png', 'label_names': ['japan', 'latin']}]
```

***备注：***

- 其中 `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。
- 默认是对 `deploy/images/PULC/language_classification/word_35404.png` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。
- 预测输出为top2的预测结果，`japan` 表示该图中文字语种识别为日语，`latin` 表示该图中文字语种识别为拉丁语。

<a name="4"></a>

## 4. 模型压缩

<a name="4.1"></a>

### 4.1 SKL-UGI 知识蒸馏

SKL-UGI 知识蒸馏是 PaddleClas 提出的一种简单有效的知识蒸馏方法，关于该方法的介绍，可以参考[SKL-UGI 知识蒸馏](@ruoyu)。

<a name="4.1.1"></a>

#### 4.1.1 教师模型训练

复用`ppcls/configs/PULC/language_classification/PPLCNet/PPLCNet_x1_0.yaml`中的超参数，训练教师模型，训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/language_classification/PPLCNet/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet101_vd
```

当前教师模型最好的权重保存在`output/ResNet101_vd/best_model.pdparams`。

**备注：** 训练ResNet101_vd模型需要的显存较多，如果机器显存不够，可以将学习率和 batch size 同时缩小一定的倍数进行训练。

<a name="4.1.2"></a>

#### 4.1.2 蒸馏训练

配置文件`ppcls/configs/PULC/language_classification/PPLCNet_x1_0_distillation.yaml`提供了`SKL-UGI知识蒸馏策略`的配置。该配置将`ResNet101_vd`当作教师模型，`PPLCNet_x1_0`当作学生模型，使用[3.2.2节](#3.2.2)中介绍的蒸馏数据作为新增的无标签数据。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/language_classification/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

当前模型最好的权重保存在`output/DistillationModel/best_model_student.pdparams`。

<a name="5"></a>

## 5. 超参搜索

在 [3.2 节](#3.2)和 [4.1 节](#4.1)所使用的超参数是根据 PaddleClas 提供的 `SHAS 超参数搜索策略` 搜索得到的，如果希望在自己的数据集上得到更好的结果，可以参考[SHAS 超参数搜索策略](#TODO)来获得更好的训练超参数。

**备注：** 此部分内容是可选内容，搜索过程需要较长的时间，您可以根据自己的硬件情况来选择执行。如果没有更换数据集，可以忽略此节内容。

<a name="6"></a>

## 6. 模型推理部署

<a name="6.1"></a>

### 6.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

当使用 Paddle Inference 推理时，加载的模型类型为 inference 模型。本案例提供了两种获得 inference 模型的方法，如果希望得到和文档相同的结果，请选择[直接下载 inference 模型](#6.1.2)的方式。

<a name="6.1.1"></a>

#### 6.1.1 基于训练得到的权重导出 inference 模型

此处，我们提供了将权重和模型转换的脚本，执行该脚本可以得到对应的 inference 模型：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/language_classification/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model_student \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_language_classification_infer
```

执行完该脚本后会在`deploy/models/`下生成`PPLCNet_x1_0_language_classification_infer`文件夹，`models` 文件夹下应有如下文件结构：

```
├── PPLCNet_x1_0_language_classification_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

**备注：** 此处的最佳权重是经过知识蒸馏后的权重路径，如果没有执行知识蒸馏的步骤，最佳模型保存在`output/PPLCNet_x1_0/best_model.pdparams`中。

<a name="6.1.2"></a>

#### 6.1.2 直接下载 inference 模型

[6.1.1 小节](#6.1.1)提供了导出 inference 模型的方法，此处也提供了该场景可以下载的 inference 模型，可以直接下载体验。

```
cd deploy/models
# 下载inference 模型并解压
wget https://paddleclas.bj.bcebos.com/models/PULC/language_classification_infer.tar && tar -xf language_classification_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── language_classification_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="6.2"></a>

### 6.2 基于 Python 预测引擎推理

<a name="6.2.1"></a>

#### 6.2.1 预测单张图像

返回 `deploy` 目录：

```
cd ../
```

运行下面的命令，对图像 `./images/PULC/language_classification/word_35404.png` 进行整图文字方向分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/language_classification/inference_language_classification.yaml
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/language_classification/inference_language_classification.yaml -o Global.use_gpu=False
```

输出结果如下。

```
word_35404.png:    class id(s): [4, 6], score(s): [0.89, 0.01], label_name(s): ['japan', 'korean']
```

其中，输出为top2的预测结果，`japan` 表示该图中文字语种为日语，`korean` 表示该图中文字语种为韩语。

<a name="6.2.2"></a>

#### 6.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/language_classification/inference_language_classification.yaml -o Global.infer_imgs="./images/PULC/language_classification/"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
word_17.png:    class id(s): [9, 4], score(s): [0.80, 0.09], label_name(s): ['latin', 'japan']
word_20.png:    class id(s): [0, 4], score(s): [0.91, 0.02], label_name(s): ['arabic', 'japan']
word_35404.png:    class id(s): [4, 6], score(s): [0.89, 0.01], label_name(s): ['japan', 'korean']
```

其中，输出为top2的预测结果，`japan` 表示该图中文字语种为日语，`latin` 表示该图中文字语种为拉丁语，`arabic` 表示该图中文字语种为阿拉伯语，`korean` 表示该图中文字语种为韩语。

<a name="6.3"></a>

### 6.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../inference_deployment/cpp_deploy.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../inference_deployment/cpp_deploy_on_windows.md)完成相应的预测库编译和模型预测工作。

<a name="6.4"></a>

### 6.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../inference_deployment/paddle_serving_deploy.md)来完成相应的部署工作。

<a name="6.5"></a>

### 6.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../inference_deployment/paddle_lite_deploy.md)来完成相应的部署工作。

<a name="6.6"></a>

### 6.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](@shuilong)来完成相应的部署工作。