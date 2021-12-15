# 配置说明

---

## 简介

本文档介绍了 PaddleClas 配置文件(`ppcls/configs/*.yaml`)中各参数的含义，以便您更快地自定义或修改超参数配置。



## 配置详解

### 目录

- [1. 分类模型](#1)
  - [1.1 全局配置(Global)](#1.1)
  - [1.2 结构(Arch)](#1.2)
  - [1.3 损失函数(Loss)](#1.3)
  - [1.4 优化器(Optimizer)](#1.4)
  - [1.5 数据读取模块(DataLoader)](#1.5)
      - [1.5.1 dataset](#1.5.1)
      - [1.5.2 sampler](#1.5.2)
      - [1.5.3 loader](#1.5.3)
  - [1.6 评估指标(Metric)](#1.6)
  - [1.7 预测](#1.7)
- [2. 蒸馏模型](#2)
  - [2.1 结构(Arch)](#2.1)
  - [2.2 损失函数(Loss)](#2.2)
  - [2.3 评估指标(Metric)](#2.3)
- [3. 识别模型](#3)
  - [3.1 结构(Arch)](#3.1)
  - [3.2 评估指标(Metric)](#3.2)
  

<a name="1"></a>
### 1.分类模型

此处以 `ResNet50_vd` 在 `ImageNet-1k` 上的训练配置为例，详解各个参数的意义。[配置路径](../../../ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml)。

<a name="1.1"></a>
#### 1.1 全局配置(Global)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| checkpoints | 断点模型路径，用于恢复训练 | null | str |
| pretrained_model | 预训练模型路径 | null | str |
| output_dir | 保存模型路径 | "./output/" | str |
| save_interval | 每隔多少个 epoch 保存模型 | 1 | int |
| eval_during_train| 是否在训练时进行评估 | True | bool |
| eval_interval | 每隔多少个 epoch 进行模型评估 | 1 | int |
| epochs | 训练总 epoch 数 |  | int |
| print_batch_step | 每隔多少个 mini-batch 打印输出 | 10 | int |
| use_visualdl | 是否是用 visualdl 可视化训练过程 | False | bool |
| image_shape | 图片大小 | [3, 224, 224] | list, shape: (3,) |
| save_inference_dir | inference 模型的保存路径 | "./inference" | str |
| eval_mode | eval 的模式 | "classification" | "retrieval" |
| to_static | 是否改为静态图模式 | False | True |
| ues_dali | 是否使用 dali 库进行图像预处理 | False | True |

**注**：`pretrained_model` 也可以填写存放预训练模型的 http 地址。

<a name="1.2"></a>
#### 1.2 结构(Arch)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| name | 模型结构名字 | ResNet50 | PaddleClas 提供的模型结构 |
| class_num | 分类数 | 1000 | int |
| pretrained | 预训练模型 | False | bool,  str |

**注**：此处的 pretrained 可以设置为 `True` 或者 `False`，也可以设置权重的路径。另外当 `Global.pretrained_model` 也设置相应路径时，此处的 `pretrained` 失效。

<a name="1.3"></a>
#### 1.3 损失函数（Loss）

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| CELoss | 交叉熵损失函数 | —— | —— |
| CELoss.weight | CELoss 在整个 Loss 中的权重 | 1.0 | float |
| CELoss.epsilon | CELoss 中 label_smooth 的 epsilon 值 | 0.1 | float, 0-1 之间 |

<a name="1.4"></a>
#### 1.4 优化器(Optimizer)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| name | 优化器方法名 | "Momentum" | "RmsProp"等其他优化器 |
| momentum | momentum 值 | 0.9 | float |
| lr.name | 学习率下降方式 | "Cosine" | "Linear"、"Piecewise"等其他下降方式 |
| lr.learning_rate | 学习率初始值 | 0.1 | float |
| lr.warmup_epoch | warmup 轮数 | 0 | int，如 5 |
| regularizer.name | 正则化方法名 | "L2" | ["L1", "L2"] |
| regularizer.coeff | 正则化系数 | 0.00007 | float |

**注**：`lr.name` 不同时，新增的参数可能也不同，如当 `lr.name=Piecewise` 时，需要添加如下参数：

```
  lr:
    name: Piecewise
    learning_rate: 0.1
    decay_epochs: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
```

添加方法及参数请查看 [learning_rate.py](../../../ppcls/optimizer/learning_rate.py)。

<a name="1.5"></a>
#### 1.5 数据读取模块(DataLoader)

<a name="1.5.1"></a>
##### 1.5.1 dataset

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| name | 读取数据的类的名字 | ImageNetDataset | VeriWild 等其他读取数据类的名字 |
| image_root | 数据集存放的路径 | ./dataset/ILSVRC2012/ | str |
| cls_label_path | 数据集标签 list | ./dataset/ILSVRC2012/train_list.txt | str |
| transform_ops | 单张图片的数据预处理 | —— | —— |
| batch_transform_ops | batch 图片的数据预处理 | —— | —— |


transform_ops 中参数的意义：

| 功能名字 | 参数名字 | 具体含义 |
|:---:|:---:|:---:|
| DecodeImage | to_rgb | 数据转 RGB |
|  | channel_first | 按 CHW 排列的图片数据 |
| RandCropImage | size | 随机裁剪 |
| RandFlipImage | | 随机翻转 |
| NormalizeImage | scale | 归一化 scale 值 |
|  | mean | 归一化均值 |
|  | std | 归一化方差 |
|  | order | 归一化顺序 |
| CropImage | size | 裁剪大小 |
| ResizeImage | resize_short | 按短边调整大小 |

batch_transform_ops 中参数的含义：

| 功能名字 | 参数名字 | 具体含义 |
|:---:|:---:|:---:|
| MixupOperator | alpha | Mixup 参数值，该值越大增强越强 |

<a name="1.5.2"></a>
##### 1.5.2 sampler

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| name | sampler 类型 | DistributedBatchSampler | DistributedRandomIdentitySampler 等其他 Sampler |
| batch_size | 批大小 | 64 | int |
| drop_last | 是否丢掉最后不够 batch-size 的数据 | False | bool |
| shuffle | 数据是否做 shuffle | True | bool |

<a name="1.5.3"></a>
##### 1.5.3 loader

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| num_workers | 数据读取线程数 | 4 | int |
| use_shared_memory | 是否使用共享内存 | True | bool |

<a name="1.6"></a>
#### 1.6 评估指标(Metric)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| TopkAcc | TopkAcc | [1, 5] | list, int |

<a name="1.7"></a>
#### 1.7 预测（Infer）

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| infer_imgs | 被 infer 的图像的地址 | docs/images/whl/demo.jpg | str |
| batch_size | 批大小 | 10 | int |
| PostProcess.name | 后处理名字 | Topk | str |
| PostProcess.topk | topk 的值 | 5 | int |
| PostProcess.class_id_map_file | class id 和名字的映射文件 | ppcls/utils/imagenet1k_label_list.txt | str |

**注**：Infer 模块的 `transforms` 的解释参考数据读取模块中的 dataset 中 `transform_ops` 的解释。

<a name="2"></a>
### 2.蒸馏模型

**注**：此处以 `MobileNetV3_large_x1_0` 在 `ImageNet-1k` 上蒸馏 `MobileNetV3_small_x1_0` 的训练配置为例，详解各个参数的意义。[配置路径](../../../ppcls/configs/ImageNet/Distillation/mv3_large_x1_0_distill_mv3_small_x1_0.yaml)。这里只介绍与分类模型有区别的参数。

<a name="2.1"></a>
#### 2.1 结构（Arch）

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| name | 模型结构名字 | DistillationModel | —— |
| class_num | 分类数 | 1000 | int |
| freeze_params_list | 冻结参数列表 | [True, False] | list |
| models | 模型列表 | [Teacher, Student] | list |
| Teacher.name | 教师模型的名字 | MobileNetV3_large_x1_0 | PaddleClas 中的模型 |
| Teacher.pretrained | 教师模型预训练权重 | True | 布尔值或者预训练权重路径 |
| Teacher.use_ssld | 教师模型预训练权重是否是 ssld 权重 | True | 布尔值 |
| infer_model_name | 被 infer 模型的类型 | Student | Teacher |

**注**：

1.list 在 yaml 中体现如下：

```
  freeze_params_list:
  - True
  - False
```
2.Student 的参数情况类似，不再赘述。

<a name="2.2"></a>
#### 2.2 损失函数(Loss)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| DistillationCELoss | 蒸馏的交叉熵损失函数 | —— | —— |
| DistillationCELoss.weight | Loss 权重 | 1.0 | float |
| DistillationCELoss.model_name_pairs |  ["Student", "Teacher"] | —— | —— |
| DistillationGTCELoss.weight | 蒸馏的模型与真实 Label 的交叉熵损失函数 | —— | —— |
| DistillationGTCELos.weight | Loss 权重 | 1.0 | float |
| DistillationCELoss.model_names | 与真实 label 作交叉熵的模型名字 | ["Student"] | —— |

<a name="2.3"></a>
#### 2.3 评估指标(Metric)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| DistillationTopkAcc | DistillationTopkAcc | 包含 model_key 和 topk 两个参数 | —— |
| DistillationTopkAcc.model_key | 被评估的模型 | "Student" | "Teacher" |
| DistillationTopkAcc.topk | Topk 的值 | [1, 5] | list, int |

**注**：`DistillationTopkAcc` 与普通 `TopkAcc` 含义相同，只是只用在蒸馏任务中。

<a name="3"></a>
### 3. 识别模型

**注**：此处以 `ResNet50` 在 `LogoDet-3k` 上的训练配置为例，详解各个参数的意义。[配置路径](../../../ppcls/configs/Logo/ResNet50_ReID.yaml)。这里只介绍与分类模型有区别的参数。

<a name="3.1"></a>
#### 3.1 结构(Arch)

|     参数名字      |         具体含义          |   默认值   |                            可选值                            |
| :---------------: | :-----------------------: | :--------: | :----------------------------------------------------------: |
|       name        |         模型结构          | "RecModel" |                         ["RecModel"]                         |
| infer_output_key  |    inference 时的输出值    | “feature”  |                    ["feature", "logits"]                     |
| infer_add_softmax |  infercne 是否添加 softmax  |    False   |                        [True, False]                         |
|     Backbone.name      |    Backbone 的名字    |      ResNet50_last_stage_stride1     | PaddleClas 提供的其他 backbone |
|     Backbone.pretrained      |   Backbone 预训练模型   |      True      | 布尔值或者预训练模型路径 |
| BackboneStopLayer.name | Backbone 中的输出层名字 |     True       | Backbone 中的特征输出层的 `full_name` |
|       Neck.name        |    网络 Neck 部分名字    |      VehicleNeck      |           需传入字典结构，Neck 网络层的具体输入参数           |
|       Neck.in_channels        |    输入 Neck 部分的维度大小    |      2048      |        与 BackboneStopLayer.name 层的大小相同           |
|       Neck.out_channels        |    输出 Neck 部分的维度大小，即特征维度大小    |      512     |        int           |
|       Head.name        |    网络 Head 部分名字    |      CircleMargin      |           Arcmargin 等           |
|       Head.embedding_size        |    特征维度大小      |      512      |           与 Neck.out_channels 保持一致           |
|       Head.class_num        |    类别数     |      3000      |           int           |
|       Head.margin        |    CircleMargin 中的 margin 值    |      0.35      |           float          |
|       Head.scale        |    CircleMargin  中的 scale 值    |      64      |           int          |

**注**：

1.在 PaddleClas 中，`Neck` 部分是 Backbone 与 embedding 层的连接部分，`Head` 部分是 embedding 层与分类层的连接部分。

2.`BackboneStopLayer.name` 的获取方式可以通过将模型可视化后获取，可视化方式可以参考 [Netron](https://github.com/lutzroeder/netron) 或者 [visualdl](https://github.com/PaddlePaddle/VisualDL)。

3.调用 `tools/export_model.py` 会将模型的权重转为 inference model，其中 `infer_add_softmax` 参数会控制是否在其后增加 `Softmax` 激活函数，代码中默认为 `True`（分类任务中最后的输出层会接 `Softmax` 激活函数），识别任务中特征层无须接激活函数，此处要设置为 `False`。

<a name="3.2"></a>
#### 3.2 评估指标(Metric)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| Recallk| 召回率 | [1, 5] | list, int |
| mAP| 平均检索精度 | None | None |
