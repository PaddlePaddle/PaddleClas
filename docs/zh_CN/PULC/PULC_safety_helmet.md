# PULC 佩戴安全帽分类模型

------

## 目录

- [1. 模型和应用场景介绍](#1)
- [2. 模型快速体验](#2)
    - [2.1 安装 paddlepaddle](#2.1)
    - [2.2 安装 paddleclas](#2.2)
    - [2.3 预测](#2.3)
- [3. 模型训练、评估和预测](#3)
    - [3.1 环境配置](#3.1)
    - [3.2 数据准备](#3.2)
      - [3.2.1 数据集来源](#3.2.1)
      - [3.2.2 数据集获取](#3.2.2)
    - [3.3 模型训练](#3.3)
    - [3.4 模型评估](#3.4)
    - [3.5 模型预测](#3.5)
- [4. 模型压缩](#4)
  - [4.1 UDML 知识蒸馏](#4.1)
    - [4.1.1 教师模型训练](#4.1.1)
    - [4.1.2 蒸馏训练](#4.1.2)
- [5. 超参搜索](#5)
- [6. 模型推理部署](#6)
  - [6.1 推理模型准备](#6.1)
    - [6.1.1 基于训练得到的权重导出 inference 模型](#6.1.1)
    - [6.1.2 直接下载 inference 模型](#6.1.2)
  - [6.2 基于 Python 预测引擎推理](#6.2)
    - [6.2.1 预测单张图像](#6.2.1)
    - [6.2.2 基于文件夹的批量预测](#6.2.2)
  - [6.3 基于 C++ 预测引擎推理](#6.3)
  - [6.4 服务化部署](#6.4)
  - [6.5 端侧部署](#6.5)
  - [6.6 Paddle2ONNX 模型转换与预测](#6.6)


<a name="1"></a>

## 1. 模型和应用场景介绍

该案例提供了用户使用 PaddleClas 的超轻量图像分类方案（PULC，Practical Ultra Lightweight image Classification）快速构建轻量级、高精度、可落地的“是否佩戴安全帽”的二分类模型。该模型可以广泛应用于如建筑施工场景、工厂车间场景、交通场景等。

下表列出了判断图片中是否佩戴安全帽的二分类模型的相关指标，前三行展现了使用 Res2Net200_vd_26w_4s，SwinTranformer_tiny 和 MobileNetV3_small_x0_35 作为 backbone 训练得到的模型的相关指标，第四行至第七行依次展现了替换 backbone 为 PPLCNet_x1_0、使用 SSLD 预训练模型、使用 SSLD 预训练模型 + EDA 策略、使用 SSLD 预训练模型 + EDA 策略 + UDML 知识蒸馏策略训练得到的模型的相关指标。

| 模型 | Tpr（%） | 延时（ms） | 存储（M） | 策略 |
|-------|-----------|----------|---------------|---------------|
| SwinTranformer_tiny  | 93.57 | 91.32  | 111 | 使用ImageNet预训练模型 |
| Res2Net200_vd_26w_4s  | 98.92 | 80.99 | 284 | 使用ImageNet预训练模型 |
| MobileNetV3_small_x0_35  | 84.83 | 2.85 | 2.6 | 使用ImageNet预训练模型 |
| PPLCNet_x1_0  | 93.27 | 2.03  | 7.1 | 使用ImageNet预训练模型 |
| PPLCNet_x1_0  | 98.16 | 2.03  | 7.1 | 使用SSLD预训练模型 |
| PPLCNet_x1_0  | 99.30 | 2.03  | 7.1 | 使用SSLD预训练模型+EDA策略|
| <b>PPLCNet_x1_0<b>  | <b>99.38<b> | <b>2.03<b>  | <b>7.1<b> | 使用SSLD预训练模型+EDA策略+UDML知识蒸馏策略|

从表中可以看出，在使用服务器端大模型作为 backbone 时，SwinTranformer_tiny 精度较低，Res2Net200_vd_26w_4s 精度较高，但服务器端大模型推理速度普遍较慢。将 backbone 替换为轻量级模型 MobileNetV3_small_x0_35 后，速度可以大幅提升，但是精度显著降低。在将 backbone 替换为 PPLCNet_x1_0 后，精度较 MobileNetV3_small_x0_35 提高约 8.5 个百分点，与此同时速度快 20% 以上。在此基础上，将 PPLCNet_x1_0 的预训练模型替换为 SSLD 预训练模型后，在对推理速度无影响的前提下，精度提升约 4.9 个百分点，进一步地使用 EDA 策略后，精度可以再提升 1.1 个百分点。此时，PPLCNet_x1_0 已经超过 Res2Net200_vd_26w_4s 模型的精度，但是速度快 70+ 倍。最后，在使用 UDML 知识蒸馏后，精度可以再提升 0.08 个百分点。下面详细介绍关于 PULC 安全帽模型的训练方法和推理部署方法。

**备注：**

* `Tpr`指标的介绍可以参考 [3.3小节](#3.3)的备注部分，延时是基于 Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz 测试得到，开启MKLDNN加速策略，线程数为10。

* 关于PP-LCNet的介绍可以参考[PP-LCNet介绍](../models/PP-LCNet.md)，相关论文可以查阅[PP-LCNet paper](https://arxiv.org/abs/2109.15099)。


<a name="2"></a>

## 2. 模型快速体验

<a name="2.1"></a>  

### 2.1 安装 paddlepaddle

- 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

- 您的机器是CPU，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

<a name="2.2"></a>  

### 2.2 安装 paddleclas

使用如下命令快速安装 paddleclas

```  
pip3 install paddleclas
```

<a name="2.3"></a>

### 2.3 预测

点击[这里](https://paddleclas.bj.bcebos.com/data/PULC/pulc_demo_imgs.zip)下载 demo 数据并解压，然后在终端中切换到相应目录。

* 使用命令行快速预测

```bash
paddleclas --model_name=safety_helmet --infer_imgs=pulc_demo_imgs/safety_helmet/safety_helmet_test_1.png
```

结果如下：
```
>>> result
class_ids: [1], scores: [0.9986255], label_names: ['unwearing_helmet'], filename: pulc_demo_imgs/safety_helmet/safety_helmet_test_1.png
Predict complete!
```

**备注**： 更换其他预测的数据时，只需要改变 `--infer_imgs=xx` 中的字段即可，支持传入整个文件夹。


* 在 Python 代码中预测
```python
import paddleclas
model = paddleclas.PaddleClas(model_name="safety_helmet")
result = model.predict(input_data="pulc_demo_imgs/safety_helmet/safety_helmet_test_1.png")
print(next(result))
```

**备注**：`model.predict()` 为可迭代对象（`generator`），因此需要使用 `next()` 函数或 `for` 循环对其迭代调用。每次调用将以 `batch_size` 为单位进行一次预测，并返回预测结果, 默认 `batch_size` 为 1，如果需要更改 `batch_size`，实例化模型时，需要指定 `batch_size`，如 `model = paddleclas.PaddleClas(model_name="safety_helmet",  batch_size=2)`, 使用默认的代码返回结果示例如下：

```
>>> result
[{'class_ids': [1], 'scores': [0.9986255], 'label_names': ['unwearing_helmet'], 'filename': 'pulc_demo_imgs/safety_helmet/safety_helmet_test_1.png'}]
```

<a name="3"></a>

## 3. 模型训练、评估和预测

<a name="3.1"></a>  

### 3.1 环境配置

* 安装：请先参考文档 [环境准备](../installation/install_paddleclas.md) 配置 PaddleClas 运行环境。

<a name="3.2"></a>

### 3.2 数据准备

<a name="3.2.1"></a>

#### 3.2.1 数据集来源

本案例中所使用的所有数据集均为开源数据，数据集基于[Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)、[hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)与[Large-scale CelebFaces Attributes (CelebA) Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)处理整合而来。

<a name="3.2.2"></a>  

#### 3.2.2 数据集获取

在公开数据集的基础上经过后处理即可得到本案例需要的数据，具体处理方法如下：

* 对于 Safety-Helmet-Wearing-Dataset 数据集：根据 bbox 标签数据，对其宽、高放大 3 倍作为 bbox 对图像进行裁剪，其中带有安全帽的图像类别为0，不戴安全帽的图像类别为1；
* 对于 hard-hat-detection 数据集：仅使用其中类别标签为 “hat” 的图像，并使用 bbox 标签进行裁剪，图像类别为0；
* 对于 CelebA 数据集：仅使用其中类别标签为 “Wearing_Hat” 的图像，并使用 bbox 标签进行裁剪，图像类别为0。

在整合上述数据后，可得到共约 15 万数据，其中戴安全帽与不戴安全帽的图像数量分别约为 2.8 万与 12.1 万，然后在两个类别上分别随机选取 0.56 万张图像作为测试集，共约 1.12 万张图像，其他约 13.8 万张图像作为训练集。

处理后的数据集部分数据可视化如下：

![](../../images/PULC/docs/safety_helmet_data_demo.jpg)

此处提供了经过上述方法处理好的数据，可以直接下载得到。

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，下载并解压安全帽场景的数据。

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/PULC/safety_helmet.tar
tar -xf safety_helmet.tar
cd ../
```

执行上述命令后，`dataset/` 下存在 `safety_helmet` 目录，该目录中具有以下数据：

```
├── images
│   ├── VOC2028_part2_001209_1.jpg
│   ├── HHD_hard_hat_workers23_1.jpg
│   ├── CelebA_077809.jpg
│   ├── ...
│   └── ...
├── train_list.txt
└── val_list.txt
```

其中，`train_list.txt` 和 `val_list.txt` 分别为训练集和验证集的标签文件，所有的图像数据在 `images/` 目录下。

**备注：**

* 关于 `train_list.txt`、`val_list.txt`的格式说明，可以参考[PaddleClas分类数据集格式说明](../data_preparation/classification_dataset.md#1-数据集格式说明) 。

<a name="3.3"></a>

### 3.3 模型训练

在 `ppcls/configs/PULC/safety_helmet/PPLCNet_x1_0.yaml` 中提供了基于该场景的训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
    -c ./ppcls/configs/PULC/safety_helmet/PPLCNet_x1_0.yaml
```

验证集的最佳指标在 `0.985-0.993` 之间（数据集较小，容易造成波动）。

**备注：**

* 此时使用的指标为Tpr，该指标描述了在假正类率（Fpr）小于某一个指标时的真正类率（Tpr），是产业中二分类问题常用的指标之一。在本案例中，Fpr 为万分之一。关于 Fpr 和 Tpr 的更多介绍，可以参考[这里](https://baike.baidu.com/item/AUC/19282953)。

* 在eval时，会打印出来当前最佳的 TprAtFpr 指标，具体地，其会打印当前的 `Fpr`、`Tpr` 值，以及当前的 `threshold`值，`Tpr` 值反映了在当前 `Fpr` 值下的召回率，该值越高，代表模型越好。`threshold` 表示当前最佳 `Fpr` 所对应的分类阈值，可用于后续模型部署落地等。

<a name="3.4"></a>

### 3.4 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/safety_helmet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```

其中 `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了训练过程中的最佳参数权重文件所在的路径，如需指定其他权重文件，只需替换对应的路径即可。

<a name="3.5"></a>

### 3.5 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/safety_helmet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```

输出结果如下：

```
[{'class_ids': [1], 'scores': [0.9524797], 'label_names': ['unwearing_helmet'], 'file_name': 'deploy/images/PULC/safety_helmet/safety_helmet_test_1.png'}]
```

**备注：**

* 这里`-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

* 默认是对 `deploy/images/PULC/safety_helmet/safety_helmet_test_1.png` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。

* 二分类默认的阈值为0.5， 如果需要指定阈值，可以重写 `Infer.PostProcess.threshold` ，如 `-o Infer.PostProcess.threshold=0.9167`，该值需要根据实际应用场景来确定，在 safety_helmet 数据集的 val 验证集上，在万分之一 Fpr 下得到的最佳 Tpr 时，该值为 0.9167。

<a name="4"></a>

## 4. 模型压缩

<a name="4.1"></a>

### 4.1 UDML 知识蒸馏

UDML 知识蒸馏是一种简单有效的知识蒸馏方法，关于该方法的介绍，可以参考[UDML 知识蒸馏](../advanced_tutorials/knowledge_distillation.md#1.2.3)。

<a name="4.1.1"></a>

#### 4.1.1 蒸馏训练

配置文件 `ppcls/configs/PULC/safety_helmet/PPLCNet_x1_0_distillation.yaml` 提供了 `UDML知识蒸馏策略` 的配置。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
    -c ./ppcls/configs/PULC/safety_helmet/PPLCNet_x1_0_distillation.yaml
```

验证集的最佳指标为 `0.990-0.993` 之间，当前模型最好的权重保存在 `output/DistillationModel/best_model_student.pdparams`。

<a name="5"></a>

## 5. 超参搜索

在 [3.2 节](#3.2)和 [4.1 节](#4.1)所使用的超参数是根据 PaddleClas 提供的 `超参数搜索策略` 搜索得到的，如果希望在自己的数据集上得到更好的结果，可以参考[超参数搜索策略](PULC_train.md#4-超参搜索)来获得更好的训练超参数。

**备注**：此部分内容是可选内容，搜索过程需要较长的时间，您可以根据自己的硬件情况来选择执行。如果没有更换数据集，可以忽略此节内容。

<a name="6"></a>

## 6. 模型推理部署

<a name="6.1"></a>

### 6.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference 可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于 Paddle Inference 推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

当使用 Paddle Inference 推理时，加载的模型类型为 inference 模型。本案例提供了两种获得 inference 模型的方法，如果希望得到和文档相同的结果，请选择[直接下载 inference 模型](#6.1.2)的方式。

<a name="6.1.1"></a>

### 6.1.1 基于训练得到的权重导出 inference 模型

此处，我们提供了将权重和模型转换的脚本，执行该脚本可以得到对应的 inference 模型：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/safety_helmet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model_student \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_safety_helmet_infer
```

执行完该脚本后会在 `deploy/models/` 下生成 `PPLCNet_x1_0_safety_helmet_infer` 目录，该目录下有如下文件结构：

```
├── PPLCNet_x1_0_safety_helmet_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

**备注：** 此处的最佳权重是经过知识蒸馏后的权重路径，如果没有执行知识蒸馏的步骤，最佳模型保存在 `output/PPLCNet_x1_0/best_model.pdparams` 中。

<a name="6.1.2"></a>

### 6.1.2 直接下载 inference 模型

[6.1.1 小节](#6.1.1)提供了导出 inference 模型的方法，此处也提供了该场景可以下载的 inference 模型，可以直接下载体验。

```
cd deploy/models
# 下载 inference 模型并解压
wget https://paddleclas.bj.bcebos.com/models/PULC/safety_helmet_infer.tar && tar -xf safety_helmet_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── safety_helmet_infer
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

运行下面的命令，对图像 `./images/PULC/safety_helmet/safety_helmet_test_1.png` 进行是否佩戴安全帽分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/safety_helmet/inference_safety_helmet.yaml
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/safety_helmet/inference_safety_helmet.yaml -o Global.use_gpu=False
```

输出结果如下。

```
safety_helmet_test_1.png:       class id(s): [1], score(s): [1.00], label_name(s): ['unwearing_helmet']
```

**备注：** 二分类默认的阈值为0.5， 如果需要指定阈值，可以重写 `Infer.PostProcess.threshold` ，如 `-o Infer.PostProcess.threshold=0.9167`，该值需要根据实际应用场景来确定，在 safety_helmet 数据集的 val 验证集上，在万分之一 Fpr 下得到的最佳 Tpr 时，该值为 0.9167。该阈值的确定方法可以参考[3.3节](#3.3)备注部分。

<a name="6.2.2"></a>  

#### 6.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/safety_helmet/inference_safety_helmet.yaml -o Global.infer_imgs="./images/PULC/safety_helmet/"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
safety_helmet_test_1.png:       class id(s): [1], score(s): [1.00], label_name(s): ['unwearing_helmet']
safety_helmet_test_2.png:       class id(s): [0], score(s): [1.00], label_name(s): ['wearing_helmet']
```

其中，`wearing_helmet` 表示该图中的人佩戴了安全帽，`unwearing_helmet` 表示该图中的人未佩戴安全帽。

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

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../../deploy/paddle2onnx/readme.md)来完成相应的部署工作。
