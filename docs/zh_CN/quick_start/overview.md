# PaddleClas 代码解析

## 目录

- [1. 整体代码和目录概览](#1)
- [2. 训练模块定义](#2)
    - [2.1 数据](#2.1)
    - [2.2 模型结构](#2.2)
    - [2.3 损失函数](#2.3)
    - [2.4 优化器和学习率衰减、权重衰减策略](#2.4)
    - [2.5 训练时评估](#2.5)
    - [2.6 模型存储](#2.6)
    - [2.7 模型裁剪与量化](#2.7)
- [3. 预测部署代码和方式](#3)

<a name="1"></a>
## 1. 整体代码和目录概览

<<<<<<< HEAD
PaddleClas 主要代码和目录结构如下
=======
## 1. 项目整体介绍
PaddleClas是一个致力于为工业界和学术界提供运用PaddlePaddle快速实现图像分类和图像识别的套件库，能够帮助开发者训练和部署性能更强的视觉模型。同时，PaddleClas提供了数个特色方案：[PULC超轻量级图像分类方案](#31-PULC实用图像分类方案)、[PP-ShiTU图像识别系统](#32-PP-ShiTu图像识别系统)、[PP系列骨干网络模型](../models/ImageNet1k/model_list.md)和[SSLD半监督知识蒸馏算法](../training/advanced/ssld.md)。
<div align="center">
<img src="https://user-images.githubusercontent.com/11568925/189267545-7a6eefa0-b4fc-4ed0-ae9d-7c6d53f59798.png"/>
<p>PaddleClas全景图</p>
</div>
>>>>>>> 45e29bc... update overview docs dir

* benchmark: 文件夹下存放了一些 shell 脚本，主要是为了测试 PaddleClas 中不同模型的速度指标，如单卡训练速度指标、多卡训练速度指标等。
* dataset：文件夹下存放数据集和用于处理数据集的脚本。脚本负责将数据集处理为适合 Dataloader 处理的格式。
* deploy：部署核心代码，文件夹存放的是部署工具，支持 python/cpp inference、Hub Serveing、Paddle Lite、Slim 离线量化等多种部署方式。
* ppcls：训练核心代码，文件夹下存放 PaddleClas 框架主体。配置文件、模型训练、评估、预测、动转静导出等具体代码实现均在这里。
* tools：训练、评估、预测、模型动转静导出的入口函数和脚本均在该文件下。
* requirements.txt 文件用于安装 PaddleClas 的依赖项。使用 pip 进行升级安装使用。
* tests：PaddleClas 模型从训练到预测的全链路测试，验证各功能是否能够正常使用。

<a name="2"></a>
## 2. 训练模块定义

深度学习模型训练模块，主要包含数据、模型结构、损失函数、优化器和学习率衰减、权重衰减策略等，以下一一解读。

<a name="2.1"></a>
### 2.1 数据

<<<<<<< HEAD
对于有监督任务来说，训练数据一般包含原始数据及其标注。
在基于单标签的图像分类任务中，原始数据指的是图像数据，而标注则是该图像数据所属的类别。
PaddleClas 中，训练时需要提供标签文件，形式如下，每一行包含一条训练样本，分别表示图片路径和类别标签，用分隔符隔开（默认为空格）。
=======
### 2.1 代码总体结构

项目代码总体结构可参考下图：
<div align="center">
  <img src="https://user-images.githubusercontent.com/108920665/195777168-f59c15e2-91d3-4893-9cf4-0f93ce8b1cb6.png"/>
<p>代码结构图</p>
</div>

以下介绍各目录代码的作用。

<a name="2.1.1"></a>

### 2.1.1 benchmark

该目录存放了用于测试PaddleClas不同模型速度指标的shell脚本，如单卡训练速度指标、多卡训练速度指标等。以下是各脚本介绍：

- prepare_data.sh:下载相应的测试数据，并配置好数据路径。
- run_benchmark.sh:执行单独一个训练测试的脚本，具体调用方式，可查看脚本注释。
- run_all.sh: 执行所有训练测试的入口脚本。

具体介绍可以[参考文档](../../../../benchmark/README.md)。

<a name="2.1.2"></a>

### 2.1.2 dataset
该目录用于存放不同的数据集。数据集文件中应当包含数据集图像、训练集标签文件、验证集标签文件、测试集标签文件；数据集标签文件使用`txt格式`保存，标签文件中每一行描述一个图像数据，包括图像地址和真值标签，中间用分隔符隔开（默认为空格），格式如下：
>>>>>>> 45e29bc... update overview docs dir

```
train/n01440764/n01440764_10026.JPEG 0
train/n01440764/n01440764_10027.JPEG 0
```

<<<<<<< HEAD
在代码 `ppcls/data/dataloader/common_dataset.py` 中，包含 `CommonDataset` 类，继承自 `paddle.io.Dataset`，
该数据集类可以通过一个键值进行索引并获取指定样本。`ImageNetDataset`, `LogoDataset`, `CommonDataset` 等数据集类都继承自这个类别

对于读入的数据，需要通过数据转换，将原始的图像数据进行转换。训练时，标准的数据预处理包含：`DecodeImage`, `RandCropImage`,
`RandFlipImage`, `NormalizeImage`, `ToCHWImage`。
在配置文件中体现如下，数据预处理主要包含在 `transforms` 字段中，以列表形式呈现，会按照顺序对数据依次做这些转换。

```yaml
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/ILSVRC2012/
      cls_label_path: ./dataset/ILSVRC2012/train_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
```
=======
更详细的数据集格式说明请参考：[图像分类任务数据集说明](../training/single_label_classification/dataset.md)和[图像识别任务数据集说明](../training/metric_learning/dataset.md)。

<a name="2.1.3"></a>

### 2.1.3 docs
该目录存放了PaddleClas项目的中英文说明文档和相关说明图，包括项目教程、方法介绍、模型介绍、应用实例介绍等。

<a name="2.1.4"></a>

### 2.1.4 ppcls
该目录存放了PaddleClas的核心代码，下面详细介绍该目录下各文件内容:

**configs**

该文件夹包含PaddleClas提供的官方配置文件，包括了对应不同模型、方法的配置，可以直接调用训练或作为训练配置的参考，详细介绍可参考：[配置文件说明](../training/config_description/basic.md)。以下简单介绍配置文件各字段功能：
|字段名|功能|
|:---:|:---:|
|Global|该字段描述整体的训练配置，包括预训练权重、预训练模型、输出地址、训练设备、训练epoch数、输入图像大小等|
|Arch|该字段描述模型的网络结构参数，构建模型时主要调用该部分参数|
|Loss|该字段描述损失函数的参数配置，包括训练和验证损失函数，损失函数类型，损失函数权重等，构建损失函数时调用|
|Optimizer|该字段描述优化器部分的参数配置，构建优化器时调用|
|DataLoader|该字段描述数据处理部分参数配置，包括训练和验证过程的不同数据集读取方式、数据采样策略、数据增广方法等|
|Metric|该字段描述评价指标，包括训练和验证过程选择的评价指标及其参数配置|

**arch**

该文件夹存放了与模型组网相关的代码，进行模型组网时根据配置文件中`Arch`字段的设置，选择对应的`骨干网络`、`Neck`、`Head`以及对应的参数设置，以下简单介绍各文件夹的作用：

|文件夹|功能|
|:---:|:---:|
|backbone|PaddleClas实现的骨干网络模型，，`__init__.py`中可以查看所有模型情况，具体骨干网络模型情况可参考：[骨干网络预训练库](../models/ImageNet1k/model_list.md)|
|gears|包含特征提取网络的 `Neck`和`Head`部分代码，在识别模型中用于对骨干网络提取的特征进行转换和处理。|-|
|distill|包含知识蒸馏相关代码，详细内容可参考：[知识蒸馏介绍](../algorithm_introduction/knowledge_distillation.md)和[知识蒸馏实战](../training/advanced/knowledge_distillation.md)|
|slim|包含模型量化相关代码，详细内容可参考[算法介绍](../algorithm_introduction/prune_quantization.md)和[使用介绍](../training/advanced/prune_quantization.md)|

**data**

该目录包含了对数据进行处理的相关代码用于构建`dataloader`，构建过程中会根据配置文件`Dataloader`字段内容，选择对应的`dataset`、`sampler`、`数据增广方式`以及对应的参数设置。以下简单介绍各文件夹作用：

|文件夹|功能|
|:---:|:---:|
|dataloader|该目录包含了不同的数据集采样方法(dataset)和不同的采样策略(sampler)|
|preprocess|该目录包含对数据的预处理和数据增广方法，包括对数据样本的处理(ops)和批数据处理方法(batch_ops)，详细介绍可参考：[数据增强实战](../training/config_description/data_augmentation.md)|
|postprocess|对模型输出结果的后处理，输出对应的类别名、置信度、预测结果等|
|utils|其他常用函数|

**optimizer**
>>>>>>> 45e29bc... update overview docs dir

PaddleClas 中也包含了 `AutoAugment`, `RandAugment` 等数据增广方法，也可以通过在配置文件中配置，从而添加到训练过程的数据预处理中。
每个数据转换的方法均以类实现，方便迁移和复用，更多的数据处理具体实现过程可以参考 `ppcls/data/preprocess/ops/` 下的代码。

对于组成一个 batch 的数据，也可以使用 mixup 或者 cutmix 等方法进行数据增广。
PaddleClas 中集成了 `MixupOperator`, `CutmixOperator`, `FmixOperator` 等基于 batch 的数据增广方法，
可以在配置文件中配置 mix 参数进行配置，更加具体的实现可以参考 `ppcls/data/preprocess/batch_ops/batch_operators.py` 。

图像分类中，数据后处理主要为 `argmax` 操作，在此不再赘述。

<a name="2.2"></a>
### 2.2 模型结构

在配置文件中，模型结构定义如下

```yaml
Arch:
  name: ResNet50
  class_num: 1000
  pretrained: False
  use_ssld: False
```

`Arch.name` 表示模型名称，`Arch.pretrained` 表示是否添加预训练模型，`Arch.use_ssld` 表示是否使用基于 `SSLD` 知识蒸馏得到的预训练模型。
所有的模型名称均在 `ppcls/arch/backbone/__init__.py` 中定义。

对应的，在 `ppcls/arch/__init__.py` 中，通过 `build_model` 方法创建模型对象。

```python
def build_model(config):
    config = copy.deepcopy(config)
    model_type = config.pop("name")
    mod = importlib.import_module(__name__)
    arch = getattr(mod, model_type)(**config)
    return arch
```

<a name="2.3"></a>
### 2.3 损失函数

PaddleClas 中，包含了 `CELoss`, `JSDivLoss`, `TripletLoss`, `CenterLoss` 等损失函数，均定义在 `ppcls/loss` 中。

在 `ppcls/loss/__init__.py` 文件中，使用 `CombinedLoss` 来构建及合并损失函数，不同训练策略中所需要的损失函数与计算方法不同，PaddleClas 在构建损失函数过程中，主要考虑了以下几个因素。

1. 是否使用 label smooth
2. 是否使用 mixup 或者 cutmix
3. 是否使用蒸馏方法进行训练
4. 是否是训练 metric learning

<<<<<<< HEAD
=======
- [服务器端C++预测](../../../deploy/cpp/readme.md)
- [分类模型服务化部署](../../../deploy/paddleserving/README.md)
- [基于PaddleHub Serving服务部署](../../../deploy/hubserving/readme.md)
- [Slim功能介绍](../../../deploy/slim/README.md)
- [端侧部署](../../../deploy/lite/readme.md)
- [paddle2onnx模型转化与预测](../../../deploy/paddle2onnx/readme.md)
- [PP-ShiTu相关](../models/PP-ShiTu/README.md)
>>>>>>> 45e29bc... update overview docs dir

用户可以在配置文件中指定损失函数的类型及权重，如在训练中添加 TripletLossV2，配置文件如下：

```yaml
Loss:
  Train:
    - CELoss:
        weight: 1.0
    - TripletLossV2:
        weight: 1.0
        margin: 0.5
```

<<<<<<< HEAD
<a name="2.4"></a>
### 2.4 优化器和学习率衰减、权重衰减策略
=======
### 2.1.6 test_tipc
该目录包含了PaddleClas项目质量监控相关的脚本，提供了一键化测试各模型的各项性能指标的功能，详细内容请参考：[飞桨训推一体全流程开发文档](../../../test_tipc/README.md)
>>>>>>> 45e29bc... update overview docs dir

图像分类任务中，`Momentum` 是一种比较常用的优化器，PaddleClas 中提供了 `Momentum` 、 `RMSProp`、`Adam` 及 `AdamW` 等几种优化器策略。

权重衰减策略是一种比较常用的正则化方法，主要用于防止模型过拟合。 PaddleClas 中提供了 `L1Decay` 和 `L2Decay` 两种权重衰减策略。

学习率衰减是图像分类任务中必不可少的精度提升训练方法，PaddleClas 目前支持 `Cosine`, `Piecewise`, `Linear` 等学习率衰减策略。

<<<<<<< HEAD
在配置文件中，优化器、权重衰减策略、学习率衰减策略可以通过以下的字段进行配置。
=======
注意：此处仅介绍整体运行逻辑，建议配合[启动训练的快速体验文档](../quick_start/quick_start_classification_new_user.md)进行代码运行逻辑部分的理解。
>>>>>>> 45e29bc... update overview docs dir

```yaml
Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Piecewise
    learning_rate: 0.1
    decay_epochs: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
  regularizer:
    name: 'L2'
    coeff: 0.0001
```

在 `ppcls/optimizer/__init__.py` 中使用 `build_optimizer` 创建优化器和学习率对象。

```python
def build_optimizer(config, epochs, step_each_epoch, parameters):
    config = copy.deepcopy(config)
    # step1 build lr
    lr = build_lr_scheduler(config.pop('lr'), epochs, step_each_epoch)
    logger.debug("build lr ({}) success..".format(lr))
    # step2 build regularization
    if 'regularizer' in config and config['regularizer'] is not None:
        reg_config = config.pop('regularizer')
        reg_name = reg_config.pop('name') + 'Decay'
        reg = getattr(paddle.regularizer, reg_name)(**reg_config)
    else:
        reg = None
    logger.debug("build regularizer ({}) success..".format(reg))
    # step3 build optimizer
    optim_name = config.pop('name')
    if 'clip_norm' in config:
        clip_norm = config.pop('clip_norm')
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    else:
        grad_clip = None
    optim = getattr(optimizer, optim_name)(learning_rate=lr,
                                           weight_decay=reg,
                                           grad_clip=grad_clip,
                                           **config)(parameters=parameters)
    logger.debug("build optimizer ({}) success..".format(optim))
    return optim, lr
```

<<<<<<< HEAD
 不同优化器和权重衰减策略均以类的形式实现，具体实现可以参考文件 `ppcls/optimizer/optimizer.py`.
 不同的学习率衰减策略可以参考文件 `ppcls/optimizer/learning_rate.py` 。
=======
设置训练过程中的配置参数和各个模块的构建参数，[./ppcls/configs](../../../ppcls/configs)中包含了PaddleClas官方提供的参考配置文件。
>>>>>>> 45e29bc... update overview docs dir

<a name="2.5"></a>
### 2.5 训练时评估

模型在训练的时候，可以设置模型保存的间隔，也可以选择每隔若干个 epoch 对验证集进行评估，
从而可以保存在验证集上精度最佳的模型。配置文件中，可以通过下面的字段进行配置。

```yaml
Global:
  save_interval: 1 # 模型保存的 epoch 间隔
  eval_during_train: True # 是否进行训练时评估
  eval_interval: 1 # 评估的 epoch 间隔
```

<<<<<<< HEAD
<a name="2.6"></a>
### 2.6 模型存储
模型存储是通过 Paddle 框架的 `paddle.save()` 函数实现的，存储的是模型的动态图版本，以字典的形式存储，便于继续训练。具体实现如下

```python
def save_model(program, model_path, epoch_id, prefix='ppcls'):
    model_path = os.path.join(model_path, str(epoch_id))
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    paddle.static.save(program, model_prefix)
    logger.info(
        logger.coloring("Already save model in {}".format(model_path), "HEADER"))
```

在保存的时候有两点需要注意：
1. 只在 0 号节点上保存模型。否则多卡训练的时候，如果所有节点都保存模型到相同的路径，
2. 则多个节点写文件时可能会发生写文件冲突，导致最终保存的模型无法被正确加载。
3. 优化器参数也需要存储，方便后续的加载断点进行训练。

=======
运行训练脚本[./tools/train.py](../../../tools/train.py)启动训练，该启动脚本首先对配置文件进行解析并调用[Engine类](../../../ppcls/engine/engine.py)进行各模块构建。
<div align="center">
  <img src="https://user-images.githubusercontent.com/108920665/196380536-d9161e04-5d69-4e24-b57f-389918830cf5.png"/>
</div>

模块构建主要调用`./ppcls`文件夹下各模块的`build函数`(位于各模块的的`__init__.py`文件)以及配置文件中对应参数进行构建，如下图在Engine类中调用[build_dataloader()函数](../../../ppcls/data/__init__.py)构建dataloader。
<div align="center">
  <img src="https://user-images.githubusercontent.com/108920665/196381203-4eb961ba-c554-49a5-87ce-a9649f96bbf7.png"/>
</div>

训练脚本[./tools/train.py](../../../tools/train.py)调用Engine类完成训练所需的各个模块构建后，会调用Engine类中的`train()`方法启动训练，该方法使用[./ppcls/engine/train/train.py](../../../ppcls/engine/train/train.py)中的`train_epoch（）函数`进行模型训练。
<div align="center">
  <img src="https://user-images.githubusercontent.com/108920665/196381856-28079311-3401-46e6-aaf2-db88c326de4c.png"/>
</div>
>>>>>>> 45e29bc... update overview docs dir

<a name="2.7"></a>
### 2.7 模型裁剪与量化
如果想对模型进行压缩训练，则通过下面字段进行配置
1.模型裁剪：

```yaml
Slim:
  prune:
    name: fpgm
    pruned_ratio: 0.3
```

2.模型量化：

```yaml
Slim:
  quant:
    name: pact
```

<<<<<<< HEAD
训练方法详见模型[裁剪量化使用介绍](../training/advanced/prune_quantization.md)，
算法介绍详见[裁剪量化算法介绍](../algorithm_introduction/prune_quantization.md)。
=======
### 3.1 PULC超轻量级图像分类方案
PULC是PaddleClas为了解决企业应用难题，让分类模型的训练和调参更加容易，总结出的实用轻量图像分类解决方案（PULC, Practical Ultra Lightweight Classification）。PULC融合了骨干网络、数据增广、蒸馏等多种前沿算法，可以自动训练得到轻量且高精度的图像分类模型。详情请参考：[PULC详细介绍](../training/PULC.md)
<div align="center">
  <img src="https://user-images.githubusercontent.com/19523330/173011854-b10fcd7a-b799-4dfd-a1cf-9504952a3c44.png"/>
<p>PULC超轻量级图像分类方案</p>
</div>
>>>>>>> 45e29bc... update overview docs dir

<a name="3"></a>
## 3. 预测部署代码和方式

<<<<<<< HEAD
* 如果希望将对分类模型进行离线量化，可以参考 [模型量化裁剪教程](../training/advanced/prune_quantization.md) 中离线量化部分。
* 如果希望在服务端使用 python 进行部署，可以参考 [python inference 预测教程](../deployment/image_classification/python.md)。
* 如果希望在服务端使用 cpp 进行部署，可以参考 [cpp inference 预测教程](../deployment/image_classification/cpp/linux.md)。
* 如果希望将分类模型部署为服务，可以参考 [hub serving 预测部署教程](../deployment/image_classification/paddle_hub.md)。
* 如果希望在移动端使用分类模型进行预测，可以参考 [PaddleLite 预测部署教程](../deployment/image_classification/paddle_lite.md)。
* 如果希望使用 whl 包对分类模型进行预测，可以参考 [whl 包预测](../deployment/image_classification/whl.md)。
=======
### 3.2 PP-ShiTu图像识别系统
PP-ShiTuV2是一个实用的轻量级通用图像识别系统，主要由主体检测、特征学习和向量检索三个模块组成。该系统从骨干网络选择和调整、损失函数的选择、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型裁剪量化多个方面，采用多种策略，对各个模块的模型进行优化，PP-ShiTuV2相比V1，Recall1提升近8个点。更多细节请参考：[PP-ShiTuV2图像识别系统](../models/PP-ShiTu/README.md)
<div align="center">
  <img src="../images/structure.jpg"/>
<p>PULC超轻量级图像分类方案</p>
</div>
>>>>>>> 45e29bc... update overview docs dir
