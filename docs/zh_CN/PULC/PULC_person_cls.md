# PaddleClas构建有人/无人分类案例

此处提供了用户使用 PaddleClas 快速构建轻量级、高精度、可落地的有人/无人的分类模型教程，主要基于有人/无人场景的数据，融合了轻量级骨干网络PPLCNet、SSLD预训练权重、EDA数据增强策略、SKL-UGI知识蒸馏策略、SHAS超参数搜索策略，得到精度高、速度快、易于部署的二分类模型。

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
        - [3.2.1.1 基于默认超参数训练轻量级模型](#3.2.1.1)
        - [3.2.1.2 基于默认超参数训练教师模型](#3.2.1.2)
        - [3.2.1.3 基于默认超参数进行蒸馏训练](#3.2.1.3)
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
wget https://paddleclas.bj.bcebos.com/models/PULC/person_cls_infer.tar && tar -xf person_cls_infer.tar
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

返回 `deploy` 目录：

```
cd ../
```

运行下面的命令，对图像 `./images/PULC/person/objects365_02035329.jpg` 进行有人/无人分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/person/inference_person_cls.yaml -o PostProcess.ThreshOutput.threshold=0.9794
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/person/inference_person_cls.yaml -o PostProcess.ThreshOutput.threshold=0.9794 -o Global.use_gpu=False
```

输出结果如下。

```
objects365_02035329.jpg:	class id(s): [1], score(s): [1.00], label_name(s): ['someone']
```


**备注：** 真实场景中往往需要在假正类率（Fpr）小于某一个指标下求真正类率（Tpr），该场景中的`val`数据集在千分之一Fpr下得到的最佳Tpr所得到的阈值为`0.9794`，故此处的`threshold`为`0.9794`。该阈值的确定方法可以参考[3.2节](#3.2)

<a name="2.2.2"></a> 

#### 2.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/person/inference_person_cls.yaml -o Global.infer_imgs="./images/PULC/person/"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
objects365_01780782.jpg:	class id(s): [0], score(s): [1.00], label_name(s): ['nobody']
objects365_02035329.jpg:	class id(s): [1], score(s): [1.00], label_name(s): ['someone']
```

其中，`someone` 表示该图里存在人，`nobody` 表示该图里不存在人。

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

* 本案例中所使用的所有数据集均为开源数据，`train`集合为[MS-COCO数据](https://cocodataset.org/#overview)的训练集的子集，`val`集合为[Object365数据](https://www.objects365.org/overview.html)的训练集的子集，`ImageNet_val`为[ImageNet数据](https://www.image-net.org/)的验证集。数据集的筛选流程可以参考[有人/无人场景数据集筛选方法]()。

<a name="3.2"></a> 

### 3.2 模型训练

<a name="3.2.1"></a> 

#### 3.2.1 基于默认超参数训练

<a name="3.2.1.1"></a> 

##### 3.2.1.1 基于默认超参数训练轻量级模型

在`ppcls/configs/PULC/person/PPLCNet/PPLCNet_x1_0.yaml`中提供了基于该场景的训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person/PPLCNet/PPLCNet_x1_0.yaml 
```

验证集的最佳指标在0.94-0.95之间（数据集较小，容易造成波动）。

**备注：** 

* 此时使用的指标为Tpr，该指标描述了在假正类率（Fpr）小于某一个指标时的真正类率（Tpr），是产业中二分类问题常用的指标之一。在本案例中，Fpr为千分之一。关于Fpr和Tpr的更多介绍，可以参考[这里](https://baike.baidu.com/item/AUC/19282953)。

* 在eval时，会打印出来当前最佳的TprAtFpr指标，具体地，其会打印当前的`Fpr`、`Tpr`值，以及当前的`threshold`值，`Tpr`值反映了在当前`Fpr`值下的召回率，该值越高，代表模型越好。`threshold` 表示当前最佳`Fpr`所对应的分类阈值，可用于后续模型部署落地等。

<a name="3.2.1.2"></a> 

##### 3.2.1.2 基于默认超参数训练教师模型

复用`ppcls/configs/PULC/person/PPLCNet/PPLCNet_x1_0.yaml`中的超参数，训练教师模型，训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person/PPLCNet/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet101_vd
```

验证集的最佳指标为0.96-0.98之间，当前教师模型最好的权重保存在`output/ResNet101_vd/best_model.pdparams`。

<a name="3.2.1.3"></a> 

##### 3.2.1.3 基于默认超参数进行蒸馏训练

配置文件`ppcls/configs/PULC/PULC/Distillation/PPLCNet_x1_0_distillation.yaml`提供了`SKL-UGI知识蒸馏策略`的配置。该配置将`ResNet101_vd`当作教师模型，`PPLCNet_x1_0`当作学生模型，使用ImageNet数据集的验证集作为新增的无标签数据。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person/Distillation/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

<a name="3.2.2"></a>

#### 3.2.2 超参数搜索训练

[3.2 小节](#3.2) 提供了在已经搜索并得到的超参数上进行了训练，此部分内容提供了搜索的过程，此过程是为了得到更好的训练超参数。

* 搜索运行脚本如下：

```shell
python tools/search_strategy.py -c ppcls/configs/StrategySearch/person.yaml
```

在`ppcls/configs/StrategySearch/person.yaml`中指定了具体的 GPU id 号和搜索配置。

* **注意**: 

* 3.1小节提供的默认配置已经经过了搜索，所以此过程不是必要的过程，如果自己的训练数据集有变化，可以尝试此过程。

* 此过程基于当前数据集在 V100 4 卡上大概需要耗时 10 小时，如果缺少机器资源，希望体验搜索过程，可以将`ppcls/configs/cls_demo/person/PPLCNet/PPLCNet_x1_0_search.yaml`中的`train_list.txt`和`val_list.txt`分别替换为`train_list.txt.debug`和`val_list.txt.debug`。替换list只是为了加速跑通整个搜索过程，由于数据量较小，其搜素的结果没有参考性。另外，搜索空间可以根据当前的机器资源来调整，如果机器资源有限，可以尝试缩小搜索空间，如果机器资源较充足，可以尝试扩大搜索空间。

* 如果此过程搜索的得到的超参数与[3.2.1小节](#3.2.1)提供的超参数不一致，主要是由于训练数据较小造成的波动导致，可以忽略。


<a name="4"></a>

## 4. 模型评估与推理


<a name="4.1"></a> 

### 4.1 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/person/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

<a name="4.2"></a> 

### 4.2 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/person/PPLCNet/PPLCNet_x1_0.yaml \
    -o Infer.infer_imgs=./dataset/person/val/objects365_01780637.jpg  \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.pretrained_model=Infer.PostProcess.threshold=0.9794
```

输出结果如下：

```
[{'class_ids': [0], 'scores': [0.9878496769815683], 'label_names': ['nobody'], 'file_name': './dataset/person/val/objects365_01780637.jpg'}]
```

**备注：** 这里的`Infer.PostProcess.threshold`的值需要根据实际场景来确定，此处的`0.9794`是在该场景中的`val`数据集在千分之一Fpr下得到的最佳Tpr所得到的。

<a name="4.3"></a> 

### 4.3 使用 inference 模型进行推理

<a name="4.3.1"></a> 

### 4.3.1 导出 inference 模型

通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/cls_demo/PULC/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_person
```
执行完该脚本后会在`deploy/models/`下生成`PPLCNet_x1_0_person`文件夹，该文件夹中的模型与 2.2 节下载的推理预测模型格式一致。

<a name="4.3.2"></a> 

### 4.3.2 基于 inference 模型推理预测
推理预测的脚本为：

```
python3.7 python/predict_cls.py -c configs/PULC/person/inference_person_cls.yaml -o Global.inference_model_dir="models/PPLCNet_x1_0_person" -o PostProcess.ThreshOutput.threshold=0.9794
```

**备注：**

- 此处的`PostProcess.ThreshOutput.threshold`由eval时的最佳`threshold`来确定。
- 更多关于推理的细节，可以参考[2.2节](#2.2)。

