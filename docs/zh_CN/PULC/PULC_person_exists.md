# PULC 有人/无人分类模型

------


## 目录

- [1. 模型和应用场景介绍](#1)
- [2. 模型快速体验](#2)
  - [2.1 环境配置](#2.1)
  - [2.2 三行命令快速体验](#2.2)  
- [3. 模型训练、评估和预测](#3)
    - [3.1 数据准备](#3.1)
      - [3.1.1 数据集来源](#3.1.1)
      - [3.1.2 数据集获取](#3.1.2)
    - [3.2 模型训练](#3.2)
    - [3.3 模型评估](#3.3)
    - [3.4 模型预测](#3.4)
- [4. 模型压缩](#4)
  - [4.1 知识蒸馏](#4.1)
    - [4.1.1 教师模型训练](#4.1.1)
    - [4.1.2 SKL-UGI知识蒸馏](#4.1.2)
  - [4.2 模型量化](#4.2)
- [5. 超参搜索](#5)
- [6. 模型推理部署](#6)
  - [6.1 推理模型准备](#6.1)
    - [6.1.1 导出 inference 模型](#6.1.1)
    - [6.1.2 直接下载 inference 模型](#6.1.2)
  - [6.2 基于 Python 预测引擎推理](#6.2)
    - [6.2.1 预测单张图像](#6.2.1)
    - [6.2.2 基于文件夹的批量预测](#6.2.2)
  - [6.3 基于 C++ 预测引擎推理](#6.3)
  - [6.4 服务化部署](#6.4)
  - [6.5 端侧部署](#6.5)
  - [6.6 Paddle2ONNX模型转换与预测](#6.6)


<a name="1"></a>

## 1. 模型和应用场景介绍

该案例提供了可以产出超轻量级二分类模型的方法。使用该方法训练得到的模型可以快速判断图片中是否有人，该模型可以广泛应用于如监控场景、人员进出管控场景、海量数据过滤场景等。

下表列出了判断图片中是否有人的二分类模型的相关指标，前两行展现了使用 SwinTranformer_tiny 和 MobileNetV3_large_x1_0 作为 backbone 训练得到的模型的相关指标，第三行至第六行依次展现了替换 backbone 为 PPLCNet_x1_0、使用 SSLD 预训练模型、使用 SSLD 预训练模型 + EDA 策略、使用 SSLD 预训练模型 + EDA 策略 + SKL-UGI 知识蒸馏策略训练得到的模型的相关指标。其中，最后一行的模型融合了前边的所有的训练策略, 即为通过 PULC 策略训练得到的模型，该模型与其他较大的模型相比，相同推理速度下拥有更高的精度，相同推理速度下拥有更高的精度。比如，与 SwinTransformer-tiny 相比，PULC 得到的模型在相同精度下，速度快 70+ 倍。训练方法和推理部署方法将在下面详细介绍。

| 模型 | Tpr（%） | 延时（ms） | 存储（M） | 策略 |
|-------|-----------|----------|---------------|---------------|
| SwinTranformer_tiny  | <b>95.69<b> | 175.52  | 107 | 使用ImageNet预训练模型 |
| MobileNetV3_large_x1_0  | <b>91.97<b> | 4.70  | 17 | 使用ImageNet预训练模型 |
| PPLCNet_x1_0  | <b>89.57<b> | 2.36  | 6.5 | 使用ImageNet预训练模型 |
| PPLCNet_x1_0  | <b>92.10<b> | 2.36  | 6.5 | 使用SSLD预训练模型 |
| PPLCNet_x1_0  | <b>93.43<b> | 2.36  | 6.5 | 使用SSLD预训练模型+EDA策略|
| <b>PPLCNet_x1_0<b>  | <b>95.60<b> | 2.36  | 6.5 | 使用SSLD预训练模型+EDA策略+SKL-UGI知识蒸馏策略|
     

**备注：** 
    
* `Tpr`指标的介绍可以参考 [3.2 小节](#3.2)的备注部分，延时是基于 Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz 测试得到，测试过程开启了 MKLDNN 加速策略，线程数为 10。
* 关于PPLCNet的介绍可以参考[PPLCNet介绍](../models/PP-LCNet.md)，相关论文可以查阅[PPLCNet paper](https://arxiv.org/abs/2109.15099)。


<a name="2"></a>

## 2. 模型快速体验

<a name="2.1"></a>  

### 2.1 环境配置

* 安装：请先参考 [Paddle 安装教程](../installation/install_paddle.md) 以及 [PaddleClas 安装教程](../installation/install_paddleclas.md) 配置 PaddleClas 运行环境。
 
<a name="2.2"></a>       

### 2.2 三行命令快速体验
 
（pip方式，待补充）

<a name="3"></a> 

## 3. 模型训练、评估和预测

<a name="3.1"></a> 

### 3.1 数据准备

<a name="3.1.1"></a> 

#### 3.1.1 数据集来源

本案例中所使用的所有数据集均为开源数据，`train` 集合为[MS-COCO 数据](https://cocodataset.org/#overview)的训练集的子集，`val` 集合为[Object365 数据](https://www.objects365.org/overview.html)的训练集的子集，`ImageNet_val` 为[ImageNet-1k 数据](https://www.image-net.org/)的验证集。

<a name="3.1.2"></a>     

#### 3.1.2 数据集获取

在公开数据集的基础上经过后处理即可得到本案例需要的数据，具体处理方法如下：

- 训练集合，本案例处理了 MS-COCO 数据训练集的标注文件，如果某张图含有“人”的标签，且这个框的面积在整张图中的比例大于 10%，即认为该张图中含有人，如果某张图中没有“人”的标签，则认为该张图中不含有人。经过处理后，得到 92964 条可用数据，其中有人的数据有 39813 条，无人的数据 53151 条。

- 验证集合，从 Object365 数据中随机抽取一小部分数据，使用在 MS-COCO 上训练得到的较好的模型预测这些数据，将预测结果和数据的标注文件取交集，将交集的结果按照得到训练集的方法筛选出验证集合。经过处理后，得到 27820 条可用数据。其中有人的数据有 2255 条，无人的数据有 25565 条。

处理后的数据集部分数据可视化如下：

![](../../images/PULC/docs/person_exists_data_demo.png)

此处提供了经过上述方法处理好的数据，可以直接下载得到。


进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，下载并解压有人/无人场景的数据。

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/PULC/person_exists.tar
tar -xf person_exists.tar
cd ../
```

执行上述命令后，`dataset/` 下存在 `person_exists` 目录，该目录中具有以下数据：

```

├── train
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
...
├── val
│   ├── objects365_01780637.jpg
│   ├── objects365_01780640.jpg
...
├── ImageNet_val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
...
├── train_list.txt
├── train_list.txt.debug
├── train_list_for_distill.txt
├── val_list.txt
└── val_list.txt.debug
```

其中 `train/` 和 `val/` 分别为训练集和验证集。`train_list.txt` 和 `val_list.txt` 分别为训练集和验证集的标签文件，`train_list.txt.debug` 和 `val_list.txt.debug` 分别为训练集和验证集的 `debug` 标签文件，其分别是 `train_list.txt` 和 `val_list.txt` 的子集，用该文件可以快速体验本案例的流程。`ImageNet_val/` 是 ImageNet-1k 的验证集，该集合和 `train` 集合的混合数据用于本案例的 `SKL-UGI知识蒸馏策略`，对应的训练标签文件为 `train_list_for_distill.txt` 。关于如何得到蒸馏的标签可以参考[知识蒸馏标签获得](@ruoyu)。


<a name="3.2"></a> 

### 3.2 模型训练 


在 `ppcls/configs/PULC/person_exists/PPLCNet_x1_0.yaml` 中提供了基于该场景的训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person_exists/PPLCNet_x1_0.yaml 
```

验证集的最佳指标在 `0.94-0.95` 之间（数据集较小，容易造成波动）。

**备注：** 

* 此时使用的指标为Tpr，该指标描述了在假正类率（Fpr）小于某一个指标时的真正类率（Tpr），是产业中二分类问题常用的指标之一。在本案例中，Fpr 为千分之一。关于 Fpr 和 Tpr 的更多介绍，可以参考[这里](https://baike.baidu.com/item/AUC/19282953)。

* 在eval时，会打印出来当前最佳的 TprAtFpr 指标，具体地，其会打印当前的 `Fpr`、`Tpr` 值，以及当前的 `threshold`值，`Tpr` 值反映了在当前 `Fpr` 值下的召回率，该值越高，代表模型越好。`threshold` 表示当前最佳 `Fpr` 所对应的分类阈值，可用于后续模型部署落地等。

<a name="3.3"></a>

### 3.3 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/person_exists/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/DPPLCNet_x1_0/best_model"
```

其中 `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

<a name="3.4"></a>

### 3.4 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/person_exists/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model_student \
    -o Global.pretrained_model=Infer.PostProcess.threshold=0.9794
```

输出结果如下：

```
[{'class_ids': [0], 'scores': [0.9878496769815683], 'label_names': ['nobody'], 'file_name': './dataset/person_exists/val/objects365_01780637.jpg'}]
```

**备注：** 

* 这里`-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。
    
* 默认是对 `deploy/images/PULC/person_exists/objects365_02035329.jpg` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。
    
* 这里的 `Infer.PostProcess.threshold` 的值需要根据实际场景来确定，此处的 `0.9794` 是在该场景中的 `val` 数据集在千分之一 Fpr 下得到的最佳 Tpr 所得到的。

<a name="4"></a>

## 4. 模型压缩

<a name="4.1"></a>

### 4.1 知识蒸馏

<a name="4.1.1"></a> 

#### 4.1.1 教师模型训练

复用 `ppcls/configs/PULC/person_exists/PPLCNet/PPLCNet_x1_0.yaml` 中的超参数，训练教师模型，训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person_exists/PPLCNet/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet101_vd
```

验证集的最佳指标为 `0.96-0.98` 之间，当前教师模型最好的权重保存在 `output/ResNet101_vd/best_model.pdparams`。

<a name="4.1.2"></a> 

####  4.1.2 SKL-UGI知识蒸馏

配置文件`ppcls/configs/PULC/person_exists/PPLCNet_x1_0_distillation.yaml`提供了`SKL-UGI知识蒸馏策略`的配置。该配置将`ResNet101_vd`当作教师模型，`PPLCNet_x1_0`当作学生模型，使用ImageNet数据集的验证集作为新增的无标签数据。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person_exists/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

验证集的最佳指标为 `0.95-0.97` 之间，当前模型最好的权重保存在 `output/DistillationModel/best_model_student.pdparams`。

<a name="4.2"></a> 

### 4.2 模型量化

PaddleClas 提供了基于 [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 的模型量化示例，量化后的模型体积更小、推理速度更快。您可以参考[模型量化教程](#TODO)来完成该模型的量化。

<a name="5"></a> 

## 5. 超参搜索

在 [3.2 节](#3.1)和 [4.1](#4.1) 节所使用的超参数是根据PaddleClas提供的 `SHAS超参数搜索策略` 搜索得到的，如果希望在自己的数据集上得到更好的结果，可以参考[SHAS超参数搜索策略](#TODO)来获得更好的训练超参数。

**备注：** 此部分内容是可选内容，搜索过程需要较长的时间，您可以根据自己的硬件情况来选择执行。如果没有更换数据集，可以忽略此节内容。

<a name="6"></a>

## 6. 模型推理部署

<a name="6.1"></a> 

### 6.1 推理模型准备

PaddlePaddle 支持使用预测引擎对 inference 模型进行预测推理。本节提供了两种得到 inference 模型的方法，如果希望得到和文档相同的结果，请选择直接下载 inference 模型的方式。

<a name="6.1.1"></a> 

### 6.1.1 导出 inference 模型

此处，我们提供了将权重和模型转换的脚本，执行该脚本可以得到对应的 inference 模型：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/person_exists/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model_student \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_person_exists_infer
```
执行完该脚本后会在 `deploy/models/` 下生成 `PPLCNet_x1_0_person_exists_infer` 文件夹，`models` 文件夹下应有如下文件结构：

```
├── PPLCNet_x1_0_person_exists_infer
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
wget https://paddleclas.bj.bcebos.com/models/PULC/person_exists_infer.tar && tar -xf person_exists_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── person_exists_infer
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

运行下面的命令，对图像 `./images/PULC/person_exists/objects365_02035329.jpg` 进行有人/无人分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/person_exists/inference_person_exists.yaml -o PostProcess.ThreshOutput.threshold=0.9794
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/person_exists/inference_person_exists.yaml -o PostProcess.ThreshOutput.threshold=0.9794 -o Global.use_gpu=False
```

输出结果如下。

```
objects365_02035329.jpg:	class id(s): [1], score(s): [1.00], label_name(s): ['someone']
```


**备注：** 真实场景中往往需要在假正类率（Fpr）小于某一个指标下求真正类率（Tpr），该场景中的 `val` 数据集在千分之一 Fpr 下得到的最佳 Tpr 所得到的阈值为 `0.9794`，故此处的 `threshold` 为 `0.9794`。该阈值的确定方法可以参考[3.2节](#3.2)备注部分。

<a name="6.2.2"></a>  

#### 6.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/person_exists/inference_person_exists.yaml -o Global.infer_imgs="./images/PULC/person_exists/"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
objects365_01780782.jpg:	class id(s): [0], score(s): [1.00], label_name(s): ['nobody']
objects365_02035329.jpg:	class id(s): [1], score(s): [1.00], label_name(s): ['someone']
```

其中，`someone` 表示该图里存在人，`nobody` 表示该图里不存在人。

<a name="6.3"></a> 

### 6.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../inference_deployment/cpp_deploy.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](inference_deployment/cpp_deploy_on_windows.md)完成相应的预测库编译和模型预测工作。

<a name="6.4"></a> 

### 6.4 服务化部署

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../inference_deployment/paddle_serving_deploy.md)来完成相应的部署工作。

<a name="6.5"></a> 

### 6.5 端侧部署

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../inference_deployment/paddle_lite_deploy.md)来
完成相应的部署工作。

<a name="6.6"></a> 

### 6.6 Paddle2ONNX模型转换与预测

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并做推理预测的示例，您可以参考[Paddle2ONNX模型转换与预测](@shuilong)来
完成相应的部署工作。
