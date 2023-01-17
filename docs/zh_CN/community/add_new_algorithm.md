# 添加新算法

- [1. 添加新算法可能改动的模块](#1)
  - [1.1 数据加载和处理](#1)
  - [1.2 网络](#2)
  - [1.3 后处理](#3)
  - [1.4 损失函数](#4)
  - [1.5 指标评估](#5)
  - [1.6 优化器](#6)
- [2. 合入套件必须新增的模块](#2)
  - [2.1 文档](#2.1)
  - [2.2 TIPC](#2.2)


<a name="1"></a>

## 1. 添加新算法可能改动的模块


<a name="1.1"></a>

### 1.1 数据加载和处理

数据加载和处理由不同的模块(module)组成，其完成了图片的读取、数据增强和label的制作。这一部分在[ppcls/data](../../../ppcls/data)下。 各个文件及文件夹作用说明如下:

```bash
ppcls/data/
├── dataloader #  数据读取、采样模块
│   ├── xxx_dataset.py # 数据读取模块
│   ├── xxx_sampler.py # 数据采样模块
│   ├── ......
├── __init__.py
├── postprocess # 模型后处理
│   ├── threshoutput.py # 对结果的卡阈值后处理
│   ├── topk.py # Topk后处理
│   ├── ......
├── preprocess # 模型前处理，通常指数据预处理
│   ├── batch_ops # batch 维度图像预处理
│   │   ├── batch_operators.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── ops # 数据增强模块
│       ├── operators.py # 常见的数据预处理
│       ├── randaugment.py # 随机增强预处理
│       └── ......
└── utils
    └── get_image_list.py # 将文件夹中的图片文件转换为img list
```

PaddleClas 内置了大量图像操作相关模块，对于没有内置的模块可通过如下步骤添加:

1. 如果只涉及单个图像的操作，在 [ppcls/data/preprocess/ops](../../../ppcls/data/preprocess/ops) 文件夹下新建文件，如果涉及整个batch的图像操作，需要在 [ppcls/data/preprocess/batch_ops](../../../ppcls/data/preprocess/batch_ops) 文件夹下新建文件，如my_module.py。
2. 在 my_module.py 文件内添加相关代码，示例代码如下:

```python
class MyModule:
    def __init__(self, *args, **kwargs):
        # your init code
        pass

    def __call__(self, img):
        # your process code
        return img
```

3. 在 [ppcls/data/preprocess/\_\_init\_\_.py](../../../ppcls/data/preprocess/__init__.py) 文件内导入添加的模块。

数据处理的所有处理步骤由不同的模块顺序执行而成，在config文件中按照列表的形式组合并执行。如:

```yaml
# angle class data process
transforms:
  - DecodeImage: # load image
      img_mode: BGR
      channel_first: False
  - MyModule:
      args1: args1
      args2: args2
```

<a name="1.2"></a>

### 1.2 网络

网络部分完成了网络的组网操作，，这一部分在[ppcls/arch/](../../../ppcls/arch/)下。 数据将按照顺序(transforms->backbone->neck->head)依次通过这四个部分。其中，非特征模型的neck和head为空。

```bash
ppcls/arch/
├── backbone
│   ├── base # PaddleClas精选模型继承的基类，负责对网络结构的自定义后处理修改
│   │   └── theseus_layer.py
│   ├── legendary_models # PaddleClas 精选的backbone
│   │   ├── pp_lcnet.py
│   │   ├── resnet.py
│   │   ├── swin_transformer.py
│   │   ├── ...
│   ├── model_zoo # PaddleClas 常见的 backbone
│   │   ├── alexnet.py
│   │   ├── efficientnet.py
│   │   ├── ......
│   └── variant_models # 模型变种模块
│       ├── pp_lcnetv2_variant.py
│       └── vgg_variant.py
├── distill # 蒸馏模块
│   └── afd_attention.py
├── gears # 识别模型的neck、head 模块
│   ├── arcmargin.py
│   └── ......
├── slim # 模型压缩模块
└── utils.py
```

PaddleClas内置了大量的常见的backbone、识别模型的neck、head模块，对于没有内置的模块可通过如下步骤添加，所有模块添加步骤相似，以backbone为例:

1. 在 [ppcls/arch/backbone/model_zoo/](../../../ppcls/arch/backbone/model_zoo/) 文件夹下新建文件，如my_backbone.py。
2. 在 my_backbone.py 文件内添加相关代码，示例代码如下:

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MyBackbone(nn.Layer):
    def __init__(self, class_num=xx, *args, **kwargs):
        super().__init__()
        # your init code
        self.class_num = class_num
        self.conv = nn.xxxx

    def forward(self, inputs):
        # your network forward
        y = self.conv(inputs)
        return y
```

3. 在 [ppcls/arch/backbone/\_\_init\_\_.py](../../../ppcls/arch/backbone/__init__.py)文件内导入添加的模块。

在完成网络的模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
# model architecture
Arch:
  name: MyBackbone
  class_num: your_class_num
```

<a name="1.3"></a>

### 1.3 后处理

后处理实现对模型结果的输出处理。这一部分在[ppcls/data/postprocess/](../../../ppcls/data/postprocess/)下。
PaddleClas内置了topk、threshoutput、attr_rec等后处理模块，对于没有内置的组件可通过如下步骤添加:

1. 在 [ppcls/data/postprocess/](../../../ppcls/data/postprocess/) 文件夹下新建文件，如 my_postprocess.py。
2. 在 my_postprocess.py 文件内添加相关代码，示例代码如下:

```python
import paddle


class MyPostProcess:
    def __init__(self, *args, **kwargs):
        # your init code
        pass

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        # you preds decode code
        preds = self.decode_preds(preds)
        if label is None:
            return preds
        # you label decode code
        label = self.decode_label(label)
        return preds, label

    def decode_preds(self, preds):
        # you preds decode code
        pass

    def decode_label(self, preds):
        # you label decode code
        pass
```

3. 在 [ppcls/data/postprocess/\_\_init\_\_.py](../../../ppcls/data/postprocess/__init__.py)文件内导入添加的模块。

在后处理模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
Infer:
  PostProcess:
    args1: args1
    args2: args2
  
```

**备注：** 该模块只在infer时使用。

<a name="1.4"></a>

### 1.4 损失函数

损失函数用于计算网络输出和label之间的距离。这一部分在[ppcls/loss/](../../../ppcls/loss/)下。
PaddleClas内置了CE Loss、BCELoss、TripletLoss等十多种损失函数，对于没有内置的模块可通过如下步骤添加:

1. 在 [ppcls/loss/](../../../ppcls/loss/) 文件夹下新建文件，如 my_loss.py。
2. 在 my_loss.py 文件内添加相关代码，示例代码如下:

```python
import paddle
from paddle import nn


class MyLoss(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        # you init code
        pass
        
    def loss(self, **kwargs):
        your loss code
        pass

    def forward(self, x, label):
        loss = self.loss(input=predicts, label=label)
        return {'your loss name': loss}
```

3. 在 [ppcls/loss/\__init\__.py](../../../ppcls/loss/__init__.py)文件内导入添加的模块。

在损失函数添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
Loss:
  Train:
    - MyLoss:
        weight: 1.0
```

**备注：** weight是指该Loss占总Loss的权重，此处支持多个Loss同时计算。

<a name="1.5"></a>

### 1.5 指标评估

指标评估用于计算网络在当前batch上的性能。这一部分在[ppcls/metric/](../../../ppcls/metric/)下。 PaddleClas内置了图像单标签分类、图像多标签分类、图像识别等算法相关的指标评估模块，对于没有内置的模块可通过如下步骤添加:

1. 在 [ppcls/metric/](../../../ppcls/metric/) 文件夹下新建文件，如my_metric.py。
2. 在 my_metric.py 文件内添加相关代码，示例代码如下:

```python

class MyMetric(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        # you init code
        pass

    def forward(self, preds, batch, *args, **kwargs):
        metric = self.get_metric(preds, batch, *args, **kwargs)
        return {'your metric name':  metric}

    def get_metric(self, *args, **kwargs):
        # you metric code
        pass

```

3. 在 [ppcls/metric/\_\_init\_\_.py](../../../ppcls/metric/__init__.py)文件内导入添加的模块。

在指标评估模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
Metric:
  Train:
    - MyMetric:
        args: args
  Eval:
    - MyMetric:
        args: args
```

<a name="1.6"></a>

### 1.6 优化器

优化器用于训练网络。优化器内部还包含了网络正则化和学习率衰减模块。 这一部分在[ppcls/optimizer/](../../ppcls/optimizer/)下。 PaddleClas内置了`Momentum`,`SGD`
,`Adam`和`AdamW`等常用的优化器模块，`Constant`、`Linear`,`Cosine`,`Step`和`Piecewise`等常用的学习率衰减模块。

对于没有内置的模块可通过如下步骤添加，以`optimizer`为例:

1. 在 [ppcls/optimizer/optimizer.py](../../../ppcls/optimizer/optimizer.py) 文件内创建自己的优化器，示例代码如下:

```python
from paddle import optimizer as optim


class MyOptim(object):
    def __init__(self, learning_rate=0.001, *args, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        # you init code
        pass
        ：
    def __call__(self, parameters):
        # It is recommended to wrap the built-in optimizer of paddle
        opt = optim.XXX(
            learning_rate=self.learning_rate,
            parameters=parameters)
        return opt

```

在优化器模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
Optimizer:
  name: MyOptim
  args1: args1
  args2: args2
  lr:
    name: Cosine
    learning_rate: 0.001
  regularizer:
    name: 'L2'
    factor: 0.0001
```

<a name="2"></a>

## 2. 合入套件必须新增的模块

<a name="2.1"></a>

### 2.1 文档

PaddleClas 中的算法都有相对应的文档说明，当给 PaddleClas 提供新的算法时，需要增加相应的文档说明。文档的位置说明如下：

算法类型|需要修改的文档地址|备注|
| --- | --- | --- |
| 骨干网络 |[文档地址1](../models/ImageNet1k)；[文档地址2](../models/ImageNet1k/README.md)|在文档地址1中新增模型介绍，在文档地址2中新增模型的精度等信息| 
| PULC |[文档地址1](../models/PULC)；[文档地址2](../models/PULC/model_list.md)|在文档地址1中新增模型介绍，在文档地址2中新增模型的精度等信息| 
| 知识蒸馏相关 |[文档地址](../training/advanced/knowledge_distillation.md)|-|
| 数据增强相关 |[文档地址1](../training/config_description/data_augmentation.md)；[文档地址2](../algorithm_introduction/data_augmentation.md)|-|
| 其他 |[文档地址1](../algorithm_introduction)；[文档地址2](../training)；[文档地址3](../models)|需要判断在文档地址1、文档地址2、文档地址3中添加相关的文档|

**备注：** 如果在添加文档过程中遇到任何问题，欢迎给我们提[issue](https://github.com/PaddlePaddle/PaddleClas/issues)。

<a name="2.2"></a>

### 2.2 TIPC

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。PaddleClas中所有模型和算法需要通过飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) ，在您提供模型时，模型需要同时通过该认证。

TIPC 当前包含很多细的方向的认证，当前只需要通过新增模型只需要通过训练和推理的基础认证即可，详情可以参考：[Linux端基础训练预测功能测试](../../../test_tipc/docs/test_train_inference_python.md)，开发流程简述如下：

- 1.新增 TIPC config，此处可以参考[DeiT](../../../test_tipc/configs/DeiT)的config配置。
- 2.走通[Linux 端基础训练预测功能测试模式一](../../../test_tipc/docs/test_train_inference_python.md#22-功能测试)，检查输出没有报错即可。

**备注：** 
- 当前只需要走通功能测试的模式一即可；
- 如果在添加TIPC过程中遇到任何问题，欢迎给我们提 [issue](https://github.com/PaddlePaddle/PaddleClas/issues)。
