# PaddleClas构建有人/无人分类案例

此处提供了用户使用 PaddleClas 快速构建轻量级、高精度、可落地的有人/无人的分类模型教程，主要基于有人/无人场景的数据，融合了轻量级骨干网络PPLCNet、SSLD预训练权重、EDA数据增强策略、KL-JS-UGI知识蒸馏策略、SHAS超参数搜索策略，得到精度高、速度快、易于部署的二分类模型。

------


## 目录

- [1. 环境配置](#1)
- [2. 有人/无人场景推理预测](#2)
  - [2.1 下载模型](#2.1)  
  - [2.2 模型推理预测](#2.2)
      - [2.2.1 预测单张图像](#2.2.1)
      - [2.2.2 基于文件夹的批量预测](#2.2.2)
- [3.有人/无人场景训练](#3)
    - [3.1 数据准备](#3.1)
    - [3.2 模型训练](#3.2)
      - [3.2.1 基于默认超参数训练](#3.2.1)
        -[3.2.1.1 基于默认超参数训练轻量级模型](#3.2.1.1)
        -[3.2.1.2 基于默认超参数训练教师模型](#3.2.1.2)
        -[3.2.1.3 基于默认超参数进行蒸馏训练](#3.2.1.3)
      - [3.2.2 超参数搜索训练](#3.2)  
- [4. 模型评估与推理](#4)
  - [4.1 模型评估](#3.1)
  - [4.2 模型预测](#3.2)
  - [4.3 使用 inference 模型进行推理](#4.3)
    - [4.3.1 导出 inference 模型](#4.3.1)
    - [4.3.2 模型推理预测](#4.3.2)
    
    
<a name="1"></a>

## 1. 环境配置

* 安装：请先参考 [Paddle 安装教程](../installation/install_paddle.md) 以及 [PaddleClas 安装教程](../installation/install_paddleclas.md) 配置 PaddleClas 运行环境。
 
<a name="2"></a> 

## 2. 有人/无人场景推理预测

<a name="2.1"></a> 

### 2.1 下载模型

* 进入 `deploy` 运行目录。

```
cd deploy
```

下载有人/无人分类的模型。

```
mkdir models
cd models
# 下载inference 模型并解压
wget https://paddleclas.bj.bcebos.com/models/cls_demo/person_cls_infer.tar && tar -xf person_cls_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── person_cls_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="2.2"></a> 

### 2.2 模型推理预测

<a name="2.2.1"></a> 

#### 2.2.1 预测单张图像

运行下面的命令，对图像 `./images/cls_demo/person/objects365_02035329.jpg` 进行有人/无人分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/cls_demo/person/inference_person_cls.yaml
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_system.py -c configs/inference_general.yaml -o Global.use_gpu=False
```

输出结果如下。

```
objects365_02035329.jpg:	class id(s): [1, 0], score(s): [1.00, 0.00], label_name(s): ['someone', 'nobody']
```

其中，`someone` 表示该图里存在人，`nobody` 表示该图里不存在人。


<a name="2.2.2"></a> 

#### 2.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_system.py -c configs/inference_general.yaml -o Global.infer_imgs="./images/cls_demo/person/"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
objects365_01780782.jpg:	class id(s): [0, 1], score(s): [1.00, 0.00], label_name(s): ['nobody', 'someone']
objects365_02035329.jpg:	class id(s): [1, 0], score(s): [1.00, 0.00], label_name(s): ['someone', 'nobody']
```

<a name="3"></a> 

## 3.有人/无人场景训练

<a name="3.1"></a> 

### 3.1 数据准备

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

<a name="3.2"></a> 

### 3.2 模型训练

<a name="3.2.1"></a> 

#### 3.2.1 基于默认超参数训练

<a name="3.2.1.1"></a> 

##### 3.2.1.1 基于默认超参数训练轻量级模型

在`ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml`中提供了基于该场景中已经搜索好的超参数，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml 
```

验证集的最佳 metric 在0.94-0.95之间（数据集较小，容易造成波动）。

<a name="3.2.1.2"></a> 

##### 3.2.1.2 基于默认超参数训练教师模型

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

<a name="3.2.1.3"></a> 

##### 3.2.1.3 基于默认超参数进行蒸馏训练

配置文件`ppcls/configs/cls_demo/person/Distillation/PPLCNet_x1_0_distillation.yaml`提供了`KL-JS-UGI知识蒸馏策略`的配置。该配置将`ResNet101_vd`当作教师模型，`PPLCNet_x1_0`当作学生模型，使用ImageNet数据集的验证集作为新增的无标签数据。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c .ppcls/configs/cls_demo/person/Distillation/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

<a name="3.2.2"></a>

#### 3.2.2 超参数搜索训练

[3.2 小节](#3.2) 提供了在已经搜索并得到的超参数上进行了训练，此部分内容提供了搜索的过程，此过程是为了得到更好的训练超参数。

* 搜索运行脚本如下：

```shell
python tools/search_strategy.py -c ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0_search.yaml
```

* **注意**: 

* 此过程基于当前数据集在 V100 4 卡上大概需要耗时 6 小时，如果缺少机器资源，希望体验搜索过程，可以将`ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0_search.yaml`中的`train_list.txt`和`val_list.txt`分别替换为`train_list.txt.debug`和`val_list.txt.debug`。替换list只是为了加速跑通整个搜索过程，由于数据量较小，其搜素的结果没有参考性。

* 如果此过程搜索的得到的超参数与3.2.1小节提供的超参数不一致，主要是由于训练数据较小造成的波动导致，可以忽略。

<a name="4"></a>

## 4. 模型评估与推理


<a name="4.1"></a> 

### 4.1 模型评估

训练好模型之后，可以通过以下命令实现对模型精度的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

<a name="4.2"></a> 

### 4.2 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml \
    -o Infer.infer_imgs=./dataset/person/val/objects365_01780637.jpg  \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```

<a name="4.3"></a> 

### 4.3 使用 inference 模型进行推理

<a name="4.3.1"></a> 

### 4.3.1 导出 inference 模型

通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_person
```
执行完该脚本后会在`deploy/models/`下生成`PPLCNet_x1_0_person`文件夹，该文件夹中的模型与 2.2 节下载的推理预测模型格式一致。

<a name="4.3.2"></a> 

### 4.3.2 基于 inference 模型推理预测
推理预测的脚本为：

```
python3.7 python/predict_cls.py -c configs/cls_demo/person/inference_person_cls.yaml -o Global.inference_model_dir="models/PPLCNet_x1_0_person"
```

更多关于推理的细节，可以参考[2.2节](#2.2)。

