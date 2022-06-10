
# 知识蒸馏实战

## 目录

- [1. 算法介绍](#1)
    - [1.1 知识蒸馏简介](#1.1)
        - [1.1.1 Response based distillation](#1.1.1)
        - [1.1.2 Feature based distillation](#1.1.2)
        - [1.1.3 Relation based distillation](#1.1.3)
    - [1.2 PaddleClas支持的知识蒸馏算法](#1.2)
        - [1.2.1 SSLD](#1.2.1)
        - [1.2.2 DML](#1.2.2)
        - [1.2.3 UDML](#1.2.3)
        - [1.2.4 AFD](#1.2.4)
        - [1.2.5 DKD](#1.2.5)
- [2. 使用方法](#2)
    - [2.1 环境配置](#2.1)
    - [2.2 数据准备](#2.2)
    - [2.3 模型训练](#2.3)
    - [2.4 模型评估](#2.4)
    - [2.5 模型预测](#2.5)
    - [2.6 模型导出与推理](#2.6)
- [3. 参考文献](#3)



<a name="1"></a>

## 1. 算法介绍

<a name="1.1"></a>

### 1.1 知识蒸馏简介

近年来，深度神经网络在计算机视觉、自然语言处理等领域被验证是一种极其有效的解决问题的方法。通过构建合适的神经网络，加以训练，最终网络模型的性能指标基本上都会超过传统算法。

在数据量足够大的情况下，通过合理构建网络模型的方式增加其参数量，可以显著改善模型性能，但是这又带来了模型复杂度急剧提升的问题。大模型在实际场景中使用的成本较高。

深度神经网络一般有较多的参数冗余，目前有几种主要的方法对模型进行压缩，减小其参数量。如裁剪、量化、知识蒸馏等，其中知识蒸馏是指使用教师模型(teacher model)去指导学生模型(student model)学习特定任务，保证小模型在参数量不变的情况下，得到比较大的性能提升，甚至获得与大模型相似的精度指标 [1]。

根据蒸馏方式的不同，可以将知识蒸馏方法分为3个不同的类别：Response based distillation、Feature based distillation、Relation based distillation。下面进行详细介绍。

<a name='1.1.1'></a>

#### 1.1.1 Response based distillation

最早的知识蒸馏算法 KD，由 Hinton 提出，训练的损失函数中除了 gt loss 之外，还引入了学生模型与教师模型输出的 KL 散度，最终精度超过单纯使用 gt loss 训练的精度。这里需要注意的是，在训练的时候，需要首先训练得到一个更大的教师模型，来指导学生模型的训练过程。

PaddleClas 中提出了一种简单使用的 SSLD 知识蒸馏算法 [6]，在训练的时候去除了对 gt label 的依赖，结合大量无标注数据，最终蒸馏训练得到的预训练模型在 15 个模型上的精度提升平均高达 3%。

上述标准的蒸馏方法是通过一个大模型作为教师模型来指导学生模型提升效果，而后来又发展出 DML(Deep Mutual Learning)互学习蒸馏方法 [7]，即通过两个结构相同的模型互相学习。具体的。相比于 KD 等依赖于大的教师模型的知识蒸馏算法，DML 脱离了对大的教师模型的依赖，蒸馏训练的流程更加简单，模型产出效率也要更高一些。

<a name='1.1.2'></a>

#### 1.1.2 Feature based distillation

Heo 等人提出了 OverHaul [8], 计算学生模型与教师模型的 feature map distance，作为蒸馏的 loss，在这里使用了学生模型、教师模型的转移，来保证二者的 feature map 可以正常地进行 distance 的计算。

基于 feature map distance 的知识蒸馏方法也能够和 `3.1 章节` 中的基于 response 的知识蒸馏算法融合在一起，同时对学生模型的输出结果和中间层 feature map 进行监督。而对于 DML 方法来说，这种融合过程更为简单，因为不需要对学生和教师模型的 feature map 进行转换，便可以完成对齐(alignment)过程。PP-OCRv2 系统中便使用了这种方法，最终大幅提升了 OCR 文字识别模型的精度。

<a name='1.1.3'></a>

#### 1.1.3 Relation based distillation

[1.1.1](#1.1.1) 和 [1.1.2](#1.1.2) 章节中的论文中主要是考虑到学生模型与教师模型的输出或者中间层 feature map，这些知识蒸馏算法只关注个体的输出结果，没有考虑到个体之间的输出关系。

Park 等人提出了 RKD [10]，基于关系的知识蒸馏算法，RKD 中进一步考虑个体输出之间的关系，使用 2 种损失函数，二阶的距离损失（distance-wise）和三阶的角度损失（angle-wise）


本论文提出的算法关系知识蒸馏（RKD）迁移教师模型得到的输出结果间的结构化关系给学生模型，不同于之前的只关注个体输出结果，RKD 算法使用两种损失函数：二阶的距离损失(distance-wise)和三阶的角度损失(angle-wise)。在最终计算蒸馏损失函数的时候，同时考虑 KD loss 和 RKD loss。最终精度优于单独使用 KD loss 蒸馏得到的模型精度。

<a name='1.2'></a>

### 1.2 PaddleClas支持的知识蒸馏算法

<a name='1.2.1'></a>

#### 1.2.1 SSLD

##### 1.2.1.1 SSLD 算法介绍

论文信息：

> [Beyond Self-Supervision: A Simple Yet Effective Network Distillation Alternative to Improve Backbones
](https://arxiv.org/abs/2103.05959)
>
> Cheng Cui, Ruoyu Guo, Yuning Du, Dongliang He, Fu Li, Zewu Wu, Qiwen Liu, Shilei Wen, Jizhou Huang, Xiaoguang Hu, Dianhai Yu, Errui Ding, Yanjun Ma
>
> arxiv, 2021

SSLD是百度于2021年提出的一种简单的半监督知识蒸馏方案，通过设计一种改进的JS散度作为损失函数，结合基于ImageNet22k数据集的数据挖掘策略，最终帮助15个骨干网络模型的精度平均提升超过3%。

更多关于SSLD的原理、模型库与使用介绍，请参考：[SSLD知识蒸馏算法介绍](./ssld.md)。


##### 1.2.1.2 SSLD 配置

SSLD配置如下所示。在模型构建Arch字段中，需要同时定义学生模型与教师模型，教师模型固定梯度，并且加载预训练参数。在损失函数Loss字段中，需要定义`DistillationDMLLoss`，作为训练的损失函数。

```yaml
# model architecture
Arch:
  name: "DistillationModel"    # 模型名称，这里使用的是蒸馏模型，
  class_num: &class_num 1000   # 类别数量，对于ImageNet1k数据集来说，类别数为1000
  pretrained_list:             # 预训练模型列表，因为在下面的子网络中指定了预训练模型，这里无需指定
  freeze_params_list:          # 固定网络参数列表，为True时，表示固定该index对应的网络
  - True
  - False
  infer_model_name: "Student"  # 在模型导出的时候，会导出Student子网络
  models:                      # 子网络列表
    - Teacher:                 # 教师模型
        name: ResNet50_vd      # 模型名称
        class_num: *class_num  # 类别数
        pretrained: True       # 预训练模型路径，如果为True，则会从官网下载默认的预训练模型
        use_ssld: True         # 是否使用SSLD蒸馏得到的预训练模型，精度会更高一些
    - Student:                 # 学生模型
        name: PPLCNet_x2_5     # 模型名称
        class_num: *class_num  # 类别数
        pretrained: False      # 预训练模型路径，可以指定为bool值或者字符串，这里为False，表示学生模型默认不加载预训练模型

# loss function config for traing/eval process
Loss:                           # 定义损失函数
  Train:                        # 定义训练的损失函数，为列表形式
    - DistillationDMLLoss:      # 蒸馏的DMLLoss，对DMLLoss进行封装，支持蒸馏结果(dict形式)的损失函数计算
        weight: 1.0             # loss权重
        model_name_pairs:       # 用于计算的模型对，这里表示计算Student和Teacher输出的损失函数
        - ["Student", "Teacher"]
  Eval:                         # 定义评估时的损失函数
    - CELoss:
        weight: 1.0
```

<a name='1.2.2'></a>

#### 1.2.2 DML

##### 1.2.2.1 DML 算法介绍

论文信息：

> [Deep Mutual Learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html)
>
> Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu
>
> CVPR, 2018

DML论文中，在蒸馏的过程中，不依赖于教师模型，两个结构相同的模型互相学习，计算彼此输出（logits）的KL散度，最终完成训练过程。


在ImageNet1k公开数据集上，效果如下所示。

| 策略 | 骨干网络 | 配置文件 | Top-1 acc | 下载链接 |
| --- | --- | --- | --- | --- |
| baseline | PPLCNet_x2_5 | [PPLCNet_x2_5.yaml](../../../ppcls/configs/ImageNet/PPLCNet/PPLCNet_x2_5.yaml) | 74.93% | - |
| DML | PPLCNet_x2_5 | [PPLCNet_x2_5_dml.yaml](../../../ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_dml.yaml) | 76.68%(**+1.75%**) | - |


* 注：完整的PPLCNet_x2_5模型训练了360epoch，这里为了方便对比，baseline和DML均训练了100epoch，因此指标比官网最终开源出来的模型精度（76.60%）低一些。


##### 1.2.2.2 DML 配置

DML配置如下所示。在模型构建Arch字段中，需要同时定义学生模型与教师模型，教师模型与学生模型均保持梯度更新状态。在损失函数Loss字段中，需要定义`DistillationDMLLoss`（学生与教师之间的JS-Div loss）以及`DistillationGTCELoss`（学生与教师关于真值标签的CE loss），作为训练的损失函数。

```yaml
Arch:
  name: "DistillationModel"
  class_num: &class_num 1000
  pretrained_list:
  freeze_params_list:        # 两个模型互相学习，因此这里两个子网络的参数均不能固定
  - False
  - False
  models:
    - Teacher:
        name: PPLCNet_x2_5   # 两个模型互学习，因此均没有加载预训练模型
        class_num: *class_num
        pretrained: False
    - Student:
        name: PPLCNet_x2_5
        class_num: *class_num
        pretrained: False

Loss:
  Train:
    - DistillationGTCELoss:    # 因为2个子网络均没有加载预训练模型，这里需要同时计算不同子网络的输出与真值标签之间的CE loss
        weight: 1.0
        model_names: ["Student", "Teacher"]
    - DistillationDMLLoss:
        weight: 1.0
        model_name_pairs:
        - ["Student", "Teacher"]
  Eval:
    - CELoss:
        weight: 1.0
```

<a name='1.2.3'></a>

#### 1.2.3 UDML

##### 1.2.3.1 UDML 算法介绍

论文信息：

UDML 是百度飞桨视觉团队提出的无需依赖教师模型的知识蒸馏算法，它基于DML进行改进，在蒸馏的过程中，除了考虑两个模型的输出信息，也考虑两个模型的中间层特征信息，从而进一步提升知识蒸馏的精度。更多关于UDML的说明与应用，请参考[PP-ShiTu论文](https://arxiv.org/abs/2111.00775)以及[PP-OCRv3论文](https://arxiv.org/abs/2109.03144)。



在ImageNet1k公开数据集上，效果如下所示。

| 策略 | 骨干网络 | 配置文件 | Top-1 acc | 下载链接 |
| --- | --- | --- | --- | --- |
| baseline | PPLCNet_x2_5 | [PPLCNet_x2_5.yaml](../../../ppcls/configs/ImageNet/PPLCNet/PPLCNet_x2_5.yaml) | 74.93% | - |
| UDML | PPLCNet_x2_5 | [PPLCNet_x2_5_dml.yaml](../../../ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_udml.yaml) | 76.74%(**+1.81%**) | - |


##### 1.2.3.2 UDML 配置


```yaml
Arch:
  name: "DistillationModel"
  class_num: &class_num 1000
  # if not null, its lengths should be same as models
  pretrained_list:
  # if not null, its lengths should be same as models
  freeze_params_list:
  - False
  - False
  models:
    - Teacher:
        name: PPLCNet_x2_5
        class_num: *class_num
        pretrained: False
        # return_patterns表示除了返回输出的logits，也会返回对应名称的中间层feature map
        return_patterns: ["blocks3", "blocks4", "blocks5", "blocks6"]
    - Student:
        name: PPLCNet_x2_5
        class_num: *class_num
        pretrained: False
        return_patterns: ["blocks3", "blocks4", "blocks5", "blocks6"]

# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
       weight: 1.0
       key: logits
       model_names: ["Student", "Teacher"]
    - DistillationDMLLoss:
        weight: 1.0
        key: logits
        model_name_pairs:
        - ["Student", "Teacher"]
    - DistillationDistanceLoss:  # 基于蒸馏结果的距离loss，这里默认使用l2 loss计算block5之间的损失函数
        weight: 1.0
        key: "blocks5"
        model_name_pairs:
        - ["Student", "Teacher"]
  Eval:
    - CELoss:
        weight: 1.0
```

**注意(：** 上述在网络中指定`return_patterns`，返回中间层特征的功能是基于TheseusLayer，更多关于TheseusLayer的使用说明，请参考：[TheseusLayer 使用说明](./theseus_layer.md)。


<a name='1.2.4'></a>

#### 1.2.4 AFD

##### 1.2.4.1 AFD 算法介绍

论文信息：


> [Show, attend and distill: Knowledge distillation via attention-based feature matching](https://arxiv.org/abs/2102.02973)
>
> Mingi Ji, Byeongho Heo, Sungrae Park
>
> AAAI, 2018

AFD提出在蒸馏的过程中，利用基于注意力的元网络学习特征之间的相对相似性，并应用识别的相似关系来控制所有可能的特征图pair的蒸馏强度。

在ImageNet1k公开数据集上，效果如下所示。

| 策略 | 骨干网络 | 配置文件 | Top-1 acc | 下载链接 |
| --- | --- | --- | --- | --- |
| baseline | ResNet18 | [ResNet18.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet18.yaml) | 70.8% | - |
| AFD | ResNet18 | [resnet34_distill_resnet18_afd.yaml](../../../ppcls/configs/ImageNet/Distillation/resnet34_distill_resnet18_afd.yaml) | 71.68%(**+0.88%**) | - |

注意：这里为了与论文的训练配置保持对齐，设置训练的迭代轮数为100epoch，因此baseline精度低于PaddleClas中开源出的模型精度（71.0%）

##### 1.2.4.2 AFD 配置

AFD配置如下所示。在模型构建Arch字段中，需要同时定义学生模型与教师模型，固定教师模型的权重。这里需要对从教师模型获取的特征进行变换，进而与学生模型进行损失函数的计算。在损失函数Loss字段中，需要定义`DistillationKLDivLoss`（学生与教师之间的KL-Div loss）、`AFDLoss`（学生与教师之间的AFD loss）以及`DistillationGTCELoss`（学生与教师关于真值标签的CE loss），作为训练的损失函数。

```yaml
Arch:
  name: "DistillationModel"
  pretrained_list:
  freeze_params_list:
  models:
    - Teacher:
        name: AttentionModel # 包含若干个串行的网络，后面的网络会将前面的网络输出作为输入并进行处理
        pretrained_list:
        freeze_params_list:
          - True
          - False
        models:
          # AttentionModel 的基础网络
          - ResNet34:
              name: ResNet34
              pretrained: True
              # return_patterns表示除了返回输出的logits，也会返回对应名称的中间层feature map
              return_patterns: &t_keys ["blocks[0]", "blocks[1]", "blocks[2]", "blocks[3]",
                                        "blocks[4]", "blocks[5]", "blocks[6]", "blocks[7]",
                                        "blocks[8]", "blocks[9]", "blocks[10]", "blocks[11]",
                                        "blocks[12]", "blocks[13]", "blocks[14]", "blocks[15]"]
          # AttentionModel的变换网络，会对基础子网络的特征进行变换  
          - LinearTransformTeacher:
              name: LinearTransformTeacher
              qk_dim: 128
              keys: *t_keys
              t_shapes: &t_shapes [[64, 56, 56], [64, 56, 56], [64, 56, 56], [128, 28, 28],
                                   [128, 28, 28], [128, 28, 28], [128, 28, 28], [256, 14, 14],
                                   [256, 14, 14], [256, 14, 14], [256, 14, 14], [256, 14, 14],
                                   [256, 14, 14], [512, 7, 7], [512, 7, 7], [512, 7, 7]]

    - Student:
        name: AttentionModel
        pretrained_list:
        freeze_params_list:
          - False
          - False
        models:
          - ResNet18:
              name: ResNet18
              pretrained: False
              return_patterns: &s_keys ["blocks[0]", "blocks[1]", "blocks[2]", "blocks[3]",
                                        "blocks[4]", "blocks[5]", "blocks[6]", "blocks[7]"]
          - LinearTransformStudent:
              name: LinearTransformStudent
              qk_dim: 128
              keys: *s_keys
              s_shapes: &s_shapes [[64, 56, 56], [64, 56, 56], [128, 28, 28], [128, 28, 28],
                                   [256, 14, 14], [256, 14, 14], [512, 7, 7], [512, 7, 7]]
              t_shapes: *t_shapes

  infer_model_name: "Student"


# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
        key: logits
    - DistillationKLDivLoss:  # 蒸馏的KL-Div loss，会根据model_name_pairs中的模型名称去提取对应模型的输出特征，计算loss
        weight: 0.9           # 该loss的权重
        model_name_pairs: [["Student", "Teacher"]]
        temperature: 4
        key: logits
    - AFDLoss:                # AFD loss
        weight: 50.0
        model_name_pair: ["Student", "Teacher"]
        student_keys: ["bilinear_key", "value"]
        teacher_keys: ["query", "value"]
        s_shapes: *s_shapes
        t_shapes: *t_shapes
  Eval:
    - CELoss:
        weight: 1.0
```

**注意(：** 上述在网络中指定`return_patterns`，返回中间层特征的功能是基于TheseusLayer，更多关于TheseusLayer的使用说明，请参考：[TheseusLayer 使用说明](./theseus_layer.md)。

<a name='1.2.5'></a>

#### 1.2.5 DKD

##### 1.2.5.1 DKD 算法介绍

论文信息：


> [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679)
>
> Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, Jiajun Liang
>
> CVPR, 2022

DKD将蒸馏中常用的 KD Loss 进行了解耦成为Target Class Knowledge Distillation(TCKD，目标类知识蒸馏)以及Non-target Class Knowledge Distillation(NCKD，非目标类知识蒸馏)两个部分，对两个部分的作用分别研究，并使它们各自的权重可以独立调节，提升了蒸馏的精度和灵活性。

在ImageNet1k公开数据集上，效果如下所示。

| 策略 | 骨干网络 | 配置文件 | Top-1 acc | 下载链接 |
| --- | --- | --- | --- | --- |
| baseline | ResNet18 | [ResNet18.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet18.yaml) | 70.8% | - |
| AFD | ResNet18 | [resnet34_distill_resnet18_dkd.yaml](../../../ppcls/configs/ImageNet/Distillation/resnet34_distill_resnet18_dkd.yaml) | 72.59%(**+1.79%**) | - |


##### 1.2.5.2 DKD 配置

DKD 配置如下所示。在模型构建Arch字段中，需要同时定义学生模型与教师模型，教师模型固定参数，且需要加载预训练模型。在损失函数Loss字段中，需要定义`DistillationDKDLoss`（学生与教师之间的DKD loss）以及`DistillationGTCELoss`（学生与教师关于真值标签的CE loss），作为训练的损失函数。


```yaml
Arch:
  name: "DistillationModel"
  # if not null, its lengths should be same as models
  pretrained_list:
  # if not null, its lengths should be same as models
  freeze_params_list:
  - True
  - False
  models:
    - Teacher:
        name: ResNet34
        pretrained: True

    - Student:
        name: ResNet18
        pretrained: False

  infer_model_name: "Student"


# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
    - DistillationDKDLoss:
        weight: 1.0
        model_name_pairs: [["Student", "Teacher"]]
        temperature: 1
        alpha: 1.0
        beta: 1.0
  Eval:
    - CELoss:
        weight: 1.0
```
<a name="2"></a>

## 2. 模型训练、评估和预测

<a name="2.1"></a>  

### 2.1 环境配置

* 安装：请先参考 [Paddle 安装教程](../installation/install_paddle.md) 以及 [PaddleClas 安装教程](../installation/install_paddleclas.md) 配置 PaddleClas 运行环境。

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
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
├── train_list.txt
...
├── val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
├── val_list.txt
```

其中 `train/` 和 `val/` 分别为训练集和验证集。`train_list.txt` 和 `val_list.txt` 分别为训练集和验证集的标签文件。


如果包含与训练集场景相似的无标注数据，则也可以按照与训练集标注完全相同的方式进行整理，将文件与当前有标注的数据集放在相同目录下，将其标签值记为0，假设整理的标签文件名为`train_list_unlabel.txt`，则可以通过下面的命令生成用于SSLD训练的标签文件。

```shell
cat train_list.txt train_list_unlabel.txt > train_list_all.txt
```


**备注：**

* 关于 `train_list.txt`、`val_list.txt`的格式说明，可以参考[PaddleClas分类数据集格式说明](../data_preparation/classification_dataset.md#1-数据集格式说明) 。


<a name="2.3"></a>

### 2.3 模型训练


以SSLD知识蒸馏算法为例，介绍知识蒸馏算法的模型训练、评估、预测等过程。配置文件为 [PPLCNet_x2_5_ssld.yaml](../../../ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml) ，使用下面的命令可以完成模型训练。


```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml
```

<a name="2.4"></a>

### 2.4 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model
```

其中 `-o Global.pretrained_model="output/DistillationModel/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

<a name="2.5"></a>

### 2.5 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model
```

输出结果如下：

```
[{'class_ids': [8, 7, 86, 82, 21], 'scores': [0.87908, 0.12091, 0.0, 0.0, 0.0], 'file_name': 'docs/images/inference_deployment/whl_demo.jpg', 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'kite']}]
```


**备注：**

* 这里`-o Global.pretrained_model="output/ResNet50/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

* 默认是对 `docs/images/inference_deployment/whl_demo.jpg` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。


<a name="2.6"></a>

### 2.6 模型导出与推理


Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

在模型推理之前需要先导出模型。对于知识蒸馏训练得到的模型，在导出时需要指定`-o Global.infer_model_name=Student`，来表示导出的模型为学生模型。具体命令如下所示。

```shell
python3 tools/export_model.py \
    -c ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml \
    -o Global.pretrained_model=./output/DistillationModel/best_model \
    -o Arch.infer_model_name=Student
```

最终在`inference`目录下会产生`inference.pdiparams`、`inference.pdiparams.info`、`inference.pdmodel` 3个文件。

关于更多模型推理相关的教程，请参考：[Python 预测推理](../inference_deployment/python_deploy.md)。


<a name="3"></a>

## 3. 参考文献

[1] Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network[J]. arXiv preprint arXiv:1503.02531, 2015.

[2] Bagherinezhad H, Horton M, Rastegari M, et al. Label refinery: Improving imagenet classification through label progression[J]. arXiv preprint arXiv:1805.02641, 2018.

[3] Yalniz I Z, Jégou H, Chen K, et al. Billion-scale semi-supervised learning for image classification[J]. arXiv preprint arXiv:1905.00546, 2019.

[4] Cubuk E D, Zoph B, Mane D, et al. Autoaugment: Learning augmentation strategies from data[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2019: 113-123.

[5] Touvron H, Vedaldi A, Douze M, et al. Fixing the train-test resolution discrepancy[C]//Advances in Neural Information Processing Systems. 2019: 8250-8260.

[6] Cui C, Guo R, Du Y, et al. Beyond Self-Supervision: A Simple Yet Effective Network Distillation Alternative to Improve Backbones[J]. arXiv preprint arXiv:2103.05959, 2021.

[7] Zhang Y, Xiang T, Hospedales T M, et al. Deep mutual learning[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4320-4328.

[8] Heo B, Kim J, Yun S, et al. A comprehensive overhaul of feature distillation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1921-1930.

[9] Du Y, Li C, Guo R, et al. PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System[J]. arXiv preprint arXiv:2109.03144, 2021.

[10] Park W, Kim D, Lu Y, et al. Relational knowledge distillation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 3967-3976.

[11] Zhao B, Cui Q, Song R, et al. Decoupled Knowledge Distillation[J]. arXiv preprint arXiv:2203.08679, 2022.

[12] Ji M, Heo B, Park S. Show, attend and distill: Knowledge distillation via attention-based feature matching[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(9): 7945-7952.
