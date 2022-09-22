# PULC 人体属性识别模型

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
  - [4.1 SKL-UGI 知识蒸馏](#4.1)
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

该案例提供了用户使用 PaddleClas 的超轻量图像分类方案（PULC，Practical Ultra Lightweight image Classification）快速构建轻量级、高精度、可落地的人体属性识别模型。该模型可以广泛应用于行人分析、行人跟踪等场景。

下表列出了不同人体属性识别模型的相关指标，前三行展现了使用 SwinTransformer_tiny、Res2Net200_vd_26w_4s 和 MobileNetV3_small_x0_35 作为 backbone 训练得到的模型的相关指标，第四行至第七行依次展现了替换 backbone 为 PPLCNet_x1_0、使用 SSLD 预训练模型、使用 SSLD 预训练模型 + EDA 策略、使用 SSLD 预训练模型 + EDA 策略 + SKL-UGI 知识蒸馏策略训练得到的模型的相关指标。


| 模型 | mA（%） | 延时（ms） | 存储（M） | 策略 |
|-------|-----------|----------|---------------|---------------|
| Res2Net200_vd_26w_4s  | 81.25 | 77.51  | 293 | 使用ImageNet预训练模型 |
| SwinTransformer_tiny  | 80.17 | 89.51  | 111 | 使用ImageNet预训练模型 |
| MobileNetV3_small_x0_35  | 70.79 | 2.90  | 1.7 | 使用ImageNet预训练模型 |
| PPLCNet_x1_0  | 76.31 | 2.01  | 7.1 | 使用ImageNet预训练模型 |
| PPLCNet_x1_0  | 77.31 | 2.01  | 7.1 | 使用SSLD预训练模型 |
| PPLCNet_x1_0  | 77.71 | 2.01  | 7.1 | 使用SSLD预训练模型+EDA策略|
| <b>PPLCNet_x1_0<b>  | <b>78.59<b> | <b>2.01<b>  | <b>7.1<b> | 使用SSLD预训练模型+EDA策略+SKL-UGI知识蒸馏策略|

从表中可以看出，backbone 为 Res2Net200_vd_26w_4s 和 SwinTransformer_tiny 时精度较高，但是推理速度较慢。将 backbone 替换为轻量级模型 MobileNetV3_small_x0_35 后，速度可以大幅提升，但是精度也大幅下降。将 backbone 替换为 PPLCNet_x1_0 时，精度较 MobileNetV3_small_x0_35 高 5.5%，于此同时，速度更快。在此基础上，使用 SSLD 预训练模型后，在不改变推理速度的前提下，精度可以提升 1%，进一步地，当融合EDA策略后，精度可以再提升 0.4%，最后，在使用 SKL-UGI 知识蒸馏后，精度可以继续提升 0.88%。此时，PPLCNet_x1_0 的精度与 SwinTransformer_tiny 仅相差1.58%，但是速度快 44 倍。关于 PULC 的训练方法和推理部署方法将在下面详细介绍。

**备注：**

* 延时是基于 Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz 测试得到，开启 MKLDNN 加速策略，线程数为10。
* 关于PP-LCNet的介绍可以参考[PP-LCNet介绍](../ImageNet1k/PP-LCNet.md)，相关论文可以查阅[PP-LCNet paper](https://arxiv.org/abs/2109.15099)。


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
paddleclas --model_name=person_attribute --infer_imgs=pulc_demo_imgs/person_attribute/090004.jpg
```

结果如下：
```
>>> result
attributes: ['Male', 'Age18-60', 'Back', 'Glasses: False', 'Hat: False', 'HoldObjectsInFront: False', 'Backpack', 'Upper: LongSleeve UpperPlaid', 'Lower:  Trousers', 'No boots'], output: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], filename: pulc_demo_imgs/person_attribute/090004.jpg
Predict complete!
```

**备注**： 更换其他预测的数据时，只需要改变 `--infer_imgs=xx` 中的字段即可，支持传入整个文件夹。


* 在 Python 代码中预测
```python
import paddleclas
model = paddleclas.PaddleClas(model_name="person_attribute")
result = model.predict(input_data="pulc_demo_imgs/person_attribute/090004.jpg")
print(next(result))
```

**备注**：`model.predict()` 为可迭代对象（`generator`），因此需要使用 `next()` 函数或 `for` 循环对其迭代调用。每次调用将以 `batch_size` 为单位进行一次预测，并返回预测结果, 默认 `batch_size` 为 1，如果需要更改 `batch_size`，实例化模型时，需要指定 `batch_size`，如 `model = paddleclas.PaddleClas(model_name="person_attribute",  batch_size=2)`, 使用默认的代码返回结果示例如下：

```
>>> result
[{'attributes': ['Male', 'Age18-60', 'Back', 'Glasses: False', 'Hat: False', 'HoldObjectsInFront: False', 'Backpack', 'Upper: LongSleeve UpperPlaid', 'Lower:  Trousers', 'No boots'], 'output': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], 'filename': 'pulc_demo_imgs/person_attribute/090004.jpg'}]
```

<a name="3"></a>

## 3. 模型训练、评估和预测

<a name="3.1"></a>  

### 3.1 环境配置

* 安装：请先参考文档 [环境准备](../../installation.md) 配置 PaddleClas 运行环境。

<a name="3.2"></a>

### 3.2 数据准备

<a name="3.2.1"></a>

#### 3.2.1 数据集来源

本案例中所使用的数据为[pa100k 数据集](https://www.v7labs.com/open-datasets/pa-100k)。

<a name="3.2.2"></a>  

#### 3.2.2 数据集获取

部分数据可视化如下所示。

<div align="center">
<img src="../../../images/PULC/docs/person_attribute_data_demo.png"  width = "500" />
</div>


我们将原始数据转换成了 PaddleClas 多标签可读的数据格式，可以直接下载。

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，下载并解压有人/无人场景的数据。

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/PULC/pa100k.tar
tar -xf pa100k.tar
cd ../
```

执行上述命令后，`dataset/` 下存在 `pa100k` 目录，该目录中具有以下数据：


执行上述命令后，`pa100k`目录中具有以下数据：

```
pa100k
├── train
│   ├── 000001.jpg
│   ├── 000002.jpg
...
├── val
│   ├── 080001.jpg
│   ├── 080002.jpg
...  
├── test
│   ├── 090001.jpg
│   ├── 090002.jpg
...
...
├── train_list.txt
├── train_val_list.txt
├── val_list.txt
├── test_list.txt
```

其中`train/`、`val/`、`test/`分别为训练集、验证集和测试集。`train_list.txt`、`val_list.txt`、`test_list.txt`分别为训练集、验证集、测试集的标签文件。在本例子中，`test_list.txt`暂时没有使用。


<a name="3.3"></a>

### 3.3 模型训练


在 `ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml` 中提供了基于该场景的训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml
```

验证集的最佳指标在 `77.71%` 左右（数据集较小，一般有0.3%左右的波动）。


<a name="3.4"></a>

### 3.4 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

其中 `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

<a name="3.5"></a>

### 3.5 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```bash
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```

输出结果如下：

```
[{'attributes': ['Male', 'Age18-60', 'Back', 'Glasses: False', 'Hat: False', 'HoldObjectsInFront: False', 'Backpack', 'Upper: LongSleeve UpperPlaid', 'Lower:  Trousers', 'No boots'], 'output': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]}]
```

**备注：**

* 这里`-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

* 默认是对 `deploy/images/PULC/person_attribute/090004.jpg` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。

<a name="4"></a>

## 4. 模型压缩

<a name="4.1"></a>

### 4.1 SKL-UGI 知识蒸馏

SKL-UGI 知识蒸馏是 PaddleClas 提出的一种简单有效的知识蒸馏方法，关于该方法的介绍，可以参考[SKL-UGI 知识蒸馏](../../training/advanced/ssld.md)。

<a name="4.1.1"></a>

#### 4.1.1 教师模型训练

复用 `ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml` 中的超参数，训练教师模型，训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet101_vd
```

验证集的最佳指标为 `80.10%` 左右，当前教师模型最好的权重保存在 `output/ResNet101_vd/best_model.pdparams`。

<a name="4.1.2"></a>

####  4.1.2 蒸馏训练

配置文件`ppcls/configs/PULC/person_attribute/PPLCNet_x1_0_Distillation.yaml`提供了`SKL-UGI知识蒸馏策略`的配置。该配置将`ResNet101_vd`当作教师模型，`PPLCNet_x1_0`当作学生模型。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0_Distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

验证集的最佳指标为 `78.5%` 左右，当前模型最好的权重保存在 `output/DistillationModel/best_model_student.pdparams`。


<a name="5"></a>

## 5. 超参搜索

在 [3.2 节](#3.2)和 [4.1 节](#4.1)所使用的超参数是根据 PaddleClas 提供的 `超参数搜索策略` 搜索得到的，如果希望在自己的数据集上得到更好的结果，可以参考[超参数搜索策略](../../training/PULC.md#4-超参搜索)来获得更好的训练超参数。

**备注：** 此部分内容是可选内容，搜索过程需要较长的时间，您可以根据自己的硬件情况来选择执行。如果没有更换数据集，可以忽略此节内容。

<a name="6"></a>

## 6. 模型推理部署

<a name="6.1"></a>

### 6.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

当使用 Paddle Inference 推理时，加载的模型类型为 inference 模型。本案例提供了两种获得 inference 模型的方法，如果希望得到和文档相同的结果，请选择[直接下载 inference 模型](#6.1.2)的方式。

<a name="6.1.1"></a>

### 6.1.1 基于训练得到的权重导出 inference 模型

此处，我们提供了将权重和模型转换的脚本，执行该脚本可以得到对应的 inference 模型：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model_student \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_person_attribute_infer
```
执行完该脚本后会在 `deploy/models/` 下生成 `PPLCNet_x1_0_person_attribute_infer` 文件夹，`models` 文件夹下应有如下文件结构：

```
├── PPLCNet_x1_0_person_attribute_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

**备注：** 此处的最佳权重是经过知识蒸馏后的权重路径，如果没有执行知识蒸馏的步骤，最佳模型保存在`output/PPLCNet_x1_0/best_model.pdparams`中。

<a name="6.1.2"></a>

### 6.1.2 直接下载 inference 模型

[6.1.1 小节](#6.1.1)提供了导出 inference 模型的方法，此处也提供了该场景可以下载的 inference 模型，可以直接下载体验。

```
cd deploy/models
# 下载 inference 模型并解压
wget https://paddleclas.bj.bcebos.com/models/PULC/person_attribute_infer.tar && tar -xf person_attribute_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── person_attribute_infer
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

运行下面的命令，对图像 `./images/PULC/person_attribute/090004.jpg` 进行行人属性识别。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/person_attribute/inference_person_attribute.yaml -o Global.use_gpu=True
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/person_attribute/inference_person_attribute.yaml -o Global.use_gpu=False
```

输出结果如下。

```
090004.jpg:     {'attributes': ['Male', 'Age18-60', 'Back', 'Glasses: False', 'Hat: False', 'HoldObjectsInFront: False', 'Backpack', 'Upper: LongSleeve UpperPlaid', 'Lower:  Trousers', 'No boots'], 'output': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]}
```

<a name="6.2.2"></a>  

#### 6.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/person_attribute/inference_person_attribute.yaml -o Global.infer_imgs="./images/PULC/person_attribute/"
```

终端中会输出该文件夹内所有图像的属性识别结果，如下所示。

```
090004.jpg:     {'attributes': ['Male', 'Age18-60', 'Back', 'Glasses: False', 'Hat: False', 'HoldObjectsInFront: False', 'Backpack', 'Upper: LongSleeve UpperPlaid', 'Lower:  Trousers', 'No boots'], 'output': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]}
090007.jpg:     {'attributes': ['Female', 'Age18-60', 'Side', 'Glasses: False', 'Hat: False', 'HoldObjectsInFront: False', 'No bag', 'Upper: ShortSleeve', 'Lower:  Skirt&Dress', 'No boots'], 'output': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]}
```

<a name="6.3"></a>

### 6.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md)完成相应的预测库编译和模型预测工作。

<a name="6.4"></a>

### 6.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../../deployment/image_classification/paddle_serving.md)来完成相应的部署工作。

<a name="6.5"></a>

### 6.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../../deployment/image_classification/paddle_lite.md)来完成相应的部署工作。

<a name="6.6"></a>

### 6.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)来完成相应的部署工作。
