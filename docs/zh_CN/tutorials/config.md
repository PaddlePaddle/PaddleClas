# 配置说明

---

## 简介

本文档介绍了PaddleClas配置文件(`configs/*.yaml`)中各参数的含义，以便您更快地自定义或修改超参数配置。

* 注意：部分参数并未在配置文件中体现，在训练或者评估时，可以直接使用`-o`进行参数的扩充或者更新，比如说`-o checkpoints=./ckp_path/ppcls`，表示在配置文件中添加（如果之前不存在）或者更新（如果之前已经包含该字段）`checkpoints`字段，其值设为`./ckp_path/ppcls`。


## 配置详解

### 基础配置

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| mode | 运行模式 | "train" | ["train"," valid"] |
| checkpoints | 断点模型路径，用于恢复训练 | "" | Str |
| last_epoch | 上一次训练结束时已经训练的epoch数量，与checkpoints一起使用 | -1 | int |
| pretrained_model | 预训练模型路径 | "" | Str |
| load_static_weights | 加载的模型是否为静态图的预训练模型 | False | bool |
| model_save_dir | 保存模型路径 | "" | Str |
| classes_num | 分类数 | 1000 | int |
| total_images | 总图片数 | 1281167 | int |
| save_interval | 每隔多少个epoch保存模型 | 1 | int |
| validate | 是否在训练时进行评估 | TRUE | bool |
| valid_interval | 每隔多少个epoch进行模型评估 | 1 | int |
| epochs | 训练总epoch数 |  | int |
| topk | 评估指标K值大小 | 5 | int |
| image_shape | 图片大小 | [3，224，224] | list, shape: (3,) |
| use_mix | 是否启用mixup | False | ['True', 'False'] |
| ls_epsilon | label_smoothing epsilon值| 0 | float |
| use_distillation | 是否进行模型蒸馏 | False | bool |

## 结构(ARCHITECTURE)

### 分类模型结构配置

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| name | 模型结构名字 | "ResNet50_vd" | PaddleClas提供的模型结构 |
| params | 模型传参 | {} | 模型结构所需的额外字典，如EfficientNet等配置文件中需要传入`padding_type`等参数，可以通过这种方式传入 |

### 识别模型结构配置

|     参数名字      |         具体含义          |   默认值   |                            可选值                            |
| :---------------: | :-----------------------: | :--------: | :----------------------------------------------------------: |
|       name        |         模型结构          | "RecModel" |                         ["RecModel"]                         |
| infer_output_key  |    inference时的输出值    | “feature”  |                    ["feature", "logits"]                     |
| infer_add_softmax |  infercne是否添加softmax  |    True    |                        [True, False]                         |
|     Backbone      |    使用Backbone的名字     |            | 需传入字典结构，包含`name`、`pretrained`等key值。其中`name`为分类模型名字， `pretrained`为布尔值 |
| BackboneStopLayer | Backbone中的feature输出层 |            | 需传入字典结构，包含`name`key值，具体值为Backbone中的特征输出层的`full_name` |
|       Neck        |    添加的网络Neck部分     |            |           需传入字典结构，Neck网络层的具体输入参数           |
|       Head        |    添加的网络Head部分     |            |           需传入字典结构，Head网络层的具体输入参数           |

### 学习率(LEARNING_RATE)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| function | decay方法名 | "Linear" | ["Linear", "Cosine", <br> "Piecewise", "CosineWarmup"] |
| params.lr | 初始学习率 | 0.1 | float |
| params.decay_epochs | piecewisedecay中<br>衰减学习率的milestone |  | list |
| params.gamma | piecewisedecay中gamma值 | 0.1 | float |
| params.warmup_epoch | warmup轮数 | 5 | int |
| parmas.steps | lineardecay衰减steps数 | 100 | int |
| params.end_lr | lineardecayend_lr值 | 0 | float |

### 优化器(OPTIMIZER)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| function | 优化器方法名 | "Momentum" | ["Momentum", "RmsProp"] |
| params.momentum | momentum值 | 0.9 | float |
| regularizer.function | 正则化方法名 | "L2" | ["L1", "L2"] |
| regularizer.factor | 正则化系数 | 0.0001 | float |

### 数据读取器与数据处理

| 参数名字 | 具体含义 |
|:---:|:---:|
| batch_size | 批大小 |
| num_workers | 数据读取器worker数量 |
| file_list | train文件列表 |
| data_dir | train文件路径 |
| shuffle_seed | 用来进行shuffle的seed值 |

数据处理

| 功能名字 | 参数名字 | 具体含义 |
|:---:|:---:|:---:|
| DecodeImage | to_rgb | 数据转RGB |
|  | to_np | 数据转numpy |
|  | channel_first | 按CHW排列的图片数据 |
| RandCropImage | size | 随机裁剪 |
| RandFlipImage | | 随机翻转 |
| NormalizeImage | scale | 归一化scale值 |
|  | mean | 归一化均值 |
|  | std | 归一化方差 |
|  | order | 归一化顺序 |
| ToCHWImage |  | 调整为CHW |
| CropImage | size | 裁剪大小 |
| ResizeImage | resize_short | 按短边调整大小 |

mix处理

| 参数名字| 具体含义|
|:---:|:---:|
| MixupOperator.alpha | mixup处理中的alpha值|
