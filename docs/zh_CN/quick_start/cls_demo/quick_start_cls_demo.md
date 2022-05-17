# PaddleClas构建有人/无人分类案例

此处提供了用户使用 PaddleClas 快速构建轻量级、高精度、可落地的有人/无人的分类模型教程，主要基于有人/无人场景的数据，融合了轻量级骨干网络PPLCNet、SSLD预训练权重、EDA数据增强策略、KL-JS-UGI知识蒸馏策略、SHAS超参数搜索策略，得到精度高、速度快、易于部署的二分类模型。

请事先参考[安装指南](../installation/install_paddleclas.md)配置运行环境和克隆 PaddleClas 代码。

------

## 目录

- [1. 数据准备](#1)
- [2. 模型训练](#2)
  - [2.1 基于搜索好的超参数训练](#2.1)
    - [2.1.1 基于搜索好的超参数训练轻量级模型](#2.1.1)
    - [2.1.2 基于搜索好的超参数训练教师模型](#2.1.2)
    - [2.1.3 基于搜索好的超参数进行蒸馏训练](#2.1.3)
  - [2.2 超参数搜索训练](2.2)
- [3. 模型评估与推理](#3)
  - [3.1 模型评估](#3.1)
  - [3.2 模型预测](#3.2)
  - [3.3 使用 inference 模型进行模型推理](#3.3)
    - [3.3.1 导出 inference 模型](#3.3.1)
    - [3.3.2 模型推理预测](#3.3.2)

<a name="1"></a>

## 1. 数据准备

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，下载并解压有人/无人场景的数据。

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/cls_demo/person.tar
tar -xf person.tar
cd ../
```

执行上述命令后，`dataset/`下存在`person`目录，该目录中具有以下数据：

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

其中`train/`和`val/`分别为训练集和验证集。`train_list.txt`和`val_list.txt`分别为训练集和验证集的标签文件，`train_list.txt.debug`和`val_list.txt.debug`分别为训练集和验证集的`debug`标签文件，其分别是`train_list.txt`和`val_list.txt`的子集，用该文件可以快速体验本案例的流程。`ImageNet_val/`是ImageNet的验证集，该集合和`train`集合的混合数据用于本案例的`KL-JS-UGI知识蒸馏策略`，对应的训练标签文件为`train_list_for_distill.txt`。

* **注意**: 

* 本案例中所使用的所有数据集均为开源数据，`train`集合为[MS-COCO数据的](https://cocodataset.org/#overview)训练集的子集，`val`集合为[Object365数据](https://www.objects365.org/overview.html)的训练集的子集，`ImageNet_val`为[ImageNet数据](https://www.image-net.org/)的验证集。

<a name="2"></a>

## 2. 模型训练

<a name="2.1"></a> 

### 2.1 基于搜索好的超参数训练

<a name="2.1.1"></a> 

#### 2.1.1 基于搜索好的超参数训练轻量级模型

在`ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml`中提供了基于该场景中已经搜索好的超参数，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml 
```

验证集的最佳 metric 在0.94-0.95之间（数据集较小，容易造成波动）。

<a name="2.1.2"></a>

#### 2.1.2 基于搜索好的超参数训练教师模型

复用`ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml`中的超参数，训练教师模型，训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet101_vd
```

验证集的最佳 metric 为0.97-0.98之间，当前教师模型最好的权重保存在`output/ResNet101_vd/best_model.pdparams`。

<a name="2.1.3"></a>

#### 2.1.3 基于搜索好的超参数进行蒸馏训练

配置文件`ppcls/configs/cls_demo/person/Distillation/PPLCNet_x1_0_distillation.yaml`提供了`KL-JS-UGI知识蒸馏策略`的配置。该配置将`ResNet101_vd`当作教师模型，`PPLCNet_x1_0`当作学生模型，使用ImageNet数据集的验证集作为新增的无标签数据。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c .ppcls/configs/cls_demo/person/Distillation/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

<a name="2.2"></a>

### 2.2 超参数搜索训练

2.1 小节提供了在已经搜索并得到的超参数上进行了训练，此部分内容提供了搜索的过程，此过程是为了得到更好的训练超参数。

* 搜索运行脚本如下：

```shell
python tools/search_strategy.py -c ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0_search.yaml
```

* **注意**: 

* 此过程基于当前数据集在 V100 4 卡上大概需要耗时 6 小时，如果缺少机器资源，希望体验搜索过程，可以将`ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0_search.yaml`中的`train_list.txt`和`val_list.txt`分别替换为`train_list.txt.debug`和`val_list.txt.debug`。替换list只是为了加速跑通整个搜索过程，由于数据量较小，其搜素的结果没有参考性。

<a name="3"></a>

## 3. 模型评估与推理


<a name="3.1"></a> 

### 3.1 模型评估

训练好模型之后，可以通过以下命令实现对模型精度的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

<a name="3.2"></a> 

### 3.2 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml \
    -o Infer.infer_imgs=./dataset/person/val/objects365_01780637.jpg  \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```

<a name="3.3"></a> 

### 3.3 使用 inference 模型进行模型推理

<a name="3.3.1"></a> 
### 3.3.1 导出 inference 模型

通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_person
```


* 默认会在 `deploy/models/PPLCNet_x1_0_person` 文件夹下生成 `inference.pdiparams`、`inference.pdmodel` 和 `inference.pdiparams.info` 文件。其中`inference.pdiparams`、`inference.pdmodel` 分别存储了模型的权重和结构，用于推理预测。


<a name="3.3.2"></a> 
### 3.3.2 模型推理预测

进入 deploy 目录下：

```bash
cd deploy
```
执行下面的命令进行预测：
```bash
python python/predict_cls.py -c configs/cls_demo/person/inference_person_cls.yaml
```

输出结果为：
```
objects365_02035329.jpg:	class id(s): [1, 0], score(s): [1.00, 0.00], label_name(s): ['someone', 'nobody']
```

如果希望预测整个文件夹的图片，可以通过`-o `来重写配置文件中的`Global.infer_imgs`字段，如预测`./images/cls_demo/person/`下所有的图片的命令为：

```bash
python python/predict_cls.py -c configs/cls_demo/person/inference_person_cls.yaml -o Global.infer_imgs=./images/cls_demo/person/
```
输出结果为：

```
objects365_01780782.jpg:	class id(s): [0, 1], score(s): [1.00, 0.00], label_name(s): ['nobody', 'someone']
objects365_02035329.jpg:	class id(s): [1, 0], score(s): [1.00, 0.00], label_name(s): ['someone', 'nobody']
```
