# PULC 含文字图像方向分类模型

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
    - [6.2.1 预测单张图片](#6.2.1)
    - [6.2.2 基于文件夹的批量预测](#6.2.2)
  - [6.3 基于 C++ 预测引擎推理](#6.3)
  - [6.4 服务化部署](#6.4)
  - [6.5 端侧部署](#6.5)
  - [6.6 Paddle2ONNX 模型转换与预测](#6.6)

<a name="1"></a>

## 1. 模型和应用场景介绍

在诸如文档扫描、证照拍摄等过程中，有时为了拍摄更清晰，会将拍摄设备进行旋转，导致得到的图片也是不同方向的。此时，标准的OCR流程无法很好地应对这些数据。利用图像分类技术，可以预先判断含文字图像的方向，并将其进行方向调整，从而提高OCR处理的准确性。该案例提供了用户使用 PaddleClas 的超轻量图像分类方案（PULC，Practical Ultra Lightweight image Classification）快速构建轻量级、高精度、可落地的含文字图像方向的分类模型。该模型可以广泛应用于金融、政务等行业的旋转图片的OCR处理场景中。

下表列出了判断含文字图像方向分类模型的相关指标，前两行展现了使用 SwinTranformer_tiny 和 MobileNetV3_small_x0_35 作为 backbone 训练得到的模型的相关指标，第三行至第五行依次展现了替换 backbone 为 PPLCNet_x1_0、使用 SSLD 预训练模型、使用EDA策略训练得到的模型的相关指标。

| 模型                    | 精度（%） | 延时（ms） | 存储（M） | 策略                       |
| ----------------------- | --------- | ---------- | --------- | -------------------------- |
| SwinTranformer_tiny     | 99.12     | 89.65      | 111       | 使用ImageNet预训练模型     |
| MobileNetV3_small_x0_35 | 83.61     | 2.95       | 2.6        | 使用ImageNet预训练模型     |
| PPLCNet_x1_0            | 97.85     | 2.16       | 7.1       | 使用ImageNet预训练模型     |
| PPLCNet_x1_0            | 99.02     | 2.16       | 7.1       | 使用SSLD预训练模型         |
| **PPLCNet_x1_0**        | **99.06** | **2.16**   | **7.1**   | 使用SSLD预训练模型+EDA策略 |

从表中可以看出，backbone 为 SwinTranformer_tiny 时精度比较高，但是推理速度较慢。将 backbone 替换为轻量级模型 MobileNetV3_small_x0_35 后，速度提升明显，但精度有了大幅下降。将 backbone 替换为 PPLCNet_x1_0 时，速度略为提升，同时精度较 MobileNetV3_small_x0_35 高了 14.24 个百分点。在此基础上，使用 SSLD 预训练模型后，在不改变推理速度的前提下，精度可以提升 1.17 个百分点，进一步地使用 EDA 策略后，精度可以再提升 0.04 个百分点。此时，PPLCNet_x1_0 与 SwinTranformer_tiny 的精度差别不大，但是速度明显变快。关于 PULC 的训练方法和推理部署方法将在下面详细介绍。

**备注：**

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
paddleclas --model_name=text_image_orientation --infer_imgs=pulc_demo_imgs/text_image_orientation/img_rot0_demo.jpg
```

结果如下：
```
>>> result
class_ids: [0, 2], scores: [0.85615, 0.05046], label_names: ['0', '180'], filename: pulc_demo_imgs/text_image_orientation/img_rot0_demo.jpg
Predict complete!
```

**备注**： 更换其他预测的数据时，只需要改变 `--infer_imgs=xx` 中的字段即可，支持传入整个文件夹。


* 在 Python 代码中预测
```python
import paddleclas
model = paddleclas.PaddleClas(model_name="text_image_orientation")
result = model.predict(input_data="pulc_demo_imgs/text_image_orientation/img_rot0_demo.jpg")
print(next(result))
```

**备注**：`model.predict()` 为可迭代对象（`generator`），因此需要使用 `next()` 函数或 `for` 循环对其迭代调用。每次调用将以 `batch_size` 为单位进行一次预测，并返回预测结果, 默认 `batch_size` 为 1，如果需要更改 `batch_size`，实例化模型时，需要指定 `batch_size`，如 `model = paddleclas.PaddleClas(model_name="text_image_orientation",  batch_size=2)`, 使用默认的代码返回结果示例如下：

```
>>> result
[{'class_ids': [0, 2], 'scores': [0.85615, 0.05046], 'label_names': ['0', '180'], 'filename': 'pulc_demo_imgs/text_image_orientation/img_rot0_demo.jpg'}]
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

[第1节](#1)中提供的模型使用内部数据训练得到，该数据集暂时不方便公开。这里基于 [ICDAR2019-ArT](https://ai.baidu.com/broad/introduction?dataset=art)、 [XFUND](https://github.com/doc-analysis/XFUND) 和 [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=introduction) 三个公开数据集构造了一个小规模含文字图像方向分类数据集，用于体验本案例。

![](../../images/PULC/docs/text_image_orientation_original_data.png)

<a name="3.2.2"></a>

#### 3.2.2 数据集获取

在公开数据集的基础上经过后处理即可得到本案例需要的数据，具体处理方法如下：

考虑到原始图片的分辨率较高，模型训练时间较长，这里将所有数据预先进行了缩放处理，在保持长宽比不变的前提下，将短边缩放到384。然后将数据进行顺时针旋转处理，分别生成90度、180度和270度的合成数据。其中，将 ICDAR2019-ArT 和 XFUND 生成的41460张数据按照 9:1 的比例随机划分成了训练集和验证集， ICDAR2015 生成的6000张数据作为`SKL-UGI知识蒸馏策略`实验中的补充数据。

处理后的数据集部分数据可视化如下：

![](../../images/PULC/docs/text_image_orientation_data_demo.png)

此处提供了经过上述方法处理好的数据，可以直接下载得到。

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，下载并解压含文字图像方向场景的数据。

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/PULC/text_image_orientation.tar
tar -xf text_image_orientation.tar
cd ../
```

执行上述命令后，`dataset/`下存在`text_image_orientation`目录，该目录中具有以下数据：

```
├── img_0
│   ├── img_rot0_0.jpg
│   ├── img_rot0_1.png
...
├── img_90
│   ├── img_rot90_0.jpg
│   ├── img_rot90_1.png
...
├── img_180
│   ├── img_rot180_0.jpg
│   ├── img_rot180_1.png
...
├── img_270
│   ├── img_rot270_0.jpg
│   ├── img_rot270_1.png
...
├── distill_data
│   ├── gt_7060_0.jpg
│   ├── gt_7060_90.jpg
...
├── train_list.txt
├── train_list.txt.debug
├── train_list_for_distill.txt
├── test_list.txt
├── test_list.txt.debug
└── label_list.txt
```

其中`img_0/`、`img_90/`、`img_180/`和`img_270/`分别存放了4个角度的训练集和验证集数据。`train_list.txt`和`test_list.txt`分别为训练集和验证集的标签文件，`train_list.txt.debug`和`test_list.txt.debug`分别为训练集和验证集的`debug`标签文件，其分别是`train_list.txt`和`test_list.txt`的子集，用该文件可以快速体验本案例的流程。`distill_data/`是补充文字数据，该集合和`train`集合的混合数据用于本案例的`SKL-UGI知识蒸馏策略`，对应的训练标签文件为`train_list_for_distill.txt`。关于如何得到蒸馏的标签可以参考[知识蒸馏标签获得](../advanced_tutorials/ssld.md#3.2)。

**备注：**

* 关于 `train_list.txt`、`val_list.txt`的格式说明，可以参考[PaddleClas分类数据集格式说明](../data_preparation/classification_dataset.md#1-数据集格式说明) 。

* 关于如何得到蒸馏的标签文件可以参考[知识蒸馏标签获得方法](../advanced_tutorials/ssld.md#3.2)。

<a name="3.3"></a>

### 3.3 模型训练

在`ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml`中提供了基于该场景的训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml
```

验证集的最佳指标在0.99左右。

**备注**：本文档中提到的训练指标均为在大规模内部数据上的训练指标，使用 demo 数据训练时，由于数据集规模较小且分布与大规模内部数据不同，无法达到该指标。可以进一步扩充自己的数据并且使用本案例中介绍的优化方法进行调优，从而达到更高的精度。

<a name="3.4"></a>

### 3.4 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

其中 `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

<a name="3.5"></a>

### 3.5 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```bash
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

输出结果如下：

```
[{'class_ids': [0, 2], 'scores': [0.85615, 0.05046], 'file_name': 'deploy/images/PULC/text_image_orientation/img_rot0_demo.jpg', 'label_names': ['0', '180']}]
```

**备注：**

- 其中 `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。
- 默认是对 `deploy/images/PULC/text_image_orientation/img_rot0_demo.jpg` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。
- 输出为top2的预测结果，`0` 表示该图文本方向为0度，`90` 表示该图文本方向为顺时针90度，`180` 表示该图文本方向为顺时针180度，`270` 表示该图文本方向为顺时针270度。

<a name="4"></a>

## 4. 模型压缩

<a name="4.1"></a>

### 4.1 SKL-UGI 知识蒸馏

SKL-UGI 知识蒸馏是 PaddleClas 提出的一种简单有效的知识蒸馏方法，关于该方法的介绍，可以参考[SKL-UGI 知识蒸馏](../advanced_tutorials/ssld.md)。

<a name="4.1.1"></a>

#### 4.1.1 教师模型训练

复用`ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml`中的超参数，训练教师模型，训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet101_vd
```

验证集的最佳指标为 0.996 左右，当前教师模型最好的权重保存在`output/ResNet101_vd/best_model.pdparams`。

**备注：** 训练 ResNet101_vd 模型需要的显存较多，如果机器显存不够，可以将学习率和 batch size 同时缩小一定的倍数进行训练。如在命令后添加以下参数 `-o DataLoader.Train.sampler.batch_size=64`, `Optimizer.lr.learning_rate=0.1`。

<a name="4.1.2"></a>

#### 4.1.2 蒸馏训练

配置文件`ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0_distillation.yaml`提供了`SKL-UGI 知识蒸馏策略`的配置。该配置将 `ResNet101_vd` 当作教师模型，`PPLCNet_x1_0` 当作学生模型，使用[3.2.2节](#3.2.2)中介绍的蒸馏数据作为新增的无标签数据。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

验证集的最佳指标为0.99左右，当前模型最好的权重保存在`output/DistillationModel/best_model_student.pdparams`。

<a name="5"></a>

## 5. 超参搜索

在 [3.2 节](#3.2)和 [4.1 节](#4.1)所使用的超参数是根据 PaddleClas 提供的 `超参数搜索策略` 搜索得到的，如果希望在自己的数据集上得到更好的结果，可以参考[超参数搜索策略](PULC_train.md#4-超参搜索)来获得更好的训练超参数。

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
    -c ./ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model_student \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_text_image_orientation_infer
```

执行完该脚本后会在`deploy/models/`下生成`PPLCNet_x1_0_text_image_orientation_infer`文件夹，`models` 文件夹下应有如下文件结构：

```
├── PPLCNet_x1_0_text_image_orientation_infer
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
wget https://paddleclas.bj.bcebos.com/models/PULC/text_image_orientation_infer.tar && tar -xf text_image_orientation_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── text_image_orientation_infer
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

运行下面的命令，对图像 `./images/PULC/text_image_orientation/img_rot0_demo.png` 进行含文字图像方向分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/text_image_orientation/inference_text_image_orientation.yaml
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/text_image_orientation/inference_text_image_orientation.yaml -o Global.use_gpu=False
```

输出结果如下。

```
img_rot0_demo.jpg:    class id(s): [0, 2], score(s): [0.86, 0.05], label_name(s): ['0', '180']
```

其中，输出为top2的预测结果，`0` 表示该图文本方向为0度，`90` 表示该图文本方向为顺时针90度，`180` 表示该图文本方向为顺时针180度，`270` 表示该图文本方向为顺时针270度。

<a name="6.2.2"></a>

#### 6.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/text_image_orientation/inference_text_image_orientation.yaml -o Global.infer_imgs="./images/PULC/text_image_orientation/"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
img_rot0_demo.jpg:    class id(s): [0, 2], score(s): [0.86, 0.05], label_name(s): ['0', '180']
img_rot180_demo.jpg:    class id(s): [2, 1], score(s): [0.88, 0.04], label_name(s): ['180', '90']
```

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
