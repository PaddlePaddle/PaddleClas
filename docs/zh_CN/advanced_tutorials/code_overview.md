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

PaddleClas 主要代码和目录结构如下

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

对于有监督任务来说，训练数据一般包含原始数据及其标注。
在基于单标签的图像分类任务中，原始数据指的是图像数据，而标注则是该图像数据所属的类别。
PaddleClas 中，训练时需要提供标签文件，形式如下，每一行包含一条训练样本，分别表示图片路径和类别标签，用分隔符隔开（默认为空格）。

```
train/n01440764/n01440764_10026.JPEG 0
train/n01440764/n01440764_10027.JPEG 0
```

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

<a name="2.4"></a>
### 2.4 优化器和学习率衰减、权重衰减策略

图像分类任务中，`Momentum` 是一种比较常用的优化器，PaddleClas 中提供了 `Momentum` 、 `RMSProp`、`Adam` 及 `AdamW` 等几种优化器策略。

权重衰减策略是一种比较常用的正则化方法，主要用于防止模型过拟合。 PaddleClas 中提供了 `L1Decay` 和 `L2Decay` 两种权重衰减策略。

学习率衰减是图像分类任务中必不可少的精度提升训练方法，PaddleClas 目前支持 `Cosine`, `Piecewise`, `Linear` 等学习率衰减策略。

在配置文件中，优化器、权重衰减策略、学习率衰减策略可以通过以下的字段进行配置。

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

 不同优化器和权重衰减策略均以类的形式实现，具体实现可以参考文件 `ppcls/optimizer/optimizer.py`.
 不同的学习率衰减策略可以参考文件 `ppcls/optimizer/learning_rate.py` 。

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

训练方法详见模型[裁剪量化使用介绍](../advanced_tutorials/model_prune_quantization.md)，
算法介绍详见[裁剪量化算法介绍](../algorithm_introduction/model_prune_quantization.md)。

<a name="3"></a>
## 3. 预测部署代码和方式

* 如果希望将对分类模型进行离线量化，可以参考 [模型量化裁剪教程](../advanced_tutorials/model_prune_quantization.md) 中离线量化部分。
* 如果希望在服务端使用 python 进行部署，可以参考 [python inference 预测教程](../inference_deployment/python_deploy.md)。
* 如果希望在服务端使用 cpp 进行部署，可以参考 [cpp inference 预测教程](../inference_deployment/cpp_deploy.md)。
* 如果希望将分类模型部署为服务，可以参考 [hub serving 预测部署教程](../inference_deployment/paddle_hub_serving_deploy.md)。
* 如果希望在移动端使用分类模型进行预测，可以参考 [PaddleLite 预测部署教程](../inference_deployment/paddle_lite_deploy.md)。
* 如果希望使用 whl 包对分类模型进行预测，可以参考 [whl 包预测](../inference_deployment/whl_deploy.md)。
