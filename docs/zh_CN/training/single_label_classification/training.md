# 图像分类
---
图像分类是根据图像的语义信息将不同类别图像区分开来，是计算机视觉中重要的基本问题，也是图像检测、图像分割、物体跟踪、行为分析等其他高层视觉任务的基础。图像分类在很多领域有广泛应用，包括安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

一般来说，图像分类通过手工特征或特征学习方法对整个图像进行全部描述，然后使用分类器判别物体类别，因此如何提取图像的特征至关重要。在深度学习算法之前使用较多的是基于词袋(Bag of Words)模型的物体分类方法。而基于深度学习的图像分类方法，可以通过有监督或无监督的方式学习层次化的特征描述，从而取代了手工设计或选择图像特征的工作。深度学习模型中的卷积神经网络(Convolution Neural Network, CNN)近年来在图像领域取得了惊人的成绩，CNN 直接利用图像像素信息作为输入，最大程度上保留了输入图像的所有信息，通过卷积操作进行特征的提取和高层抽象，模型输出直接是图像识别的结果。这种基于“输入-输出”直接端到端的学习方法取得了非常好的效果，得到了广泛的应用。

图像分类是计算机视觉里很基础但又重要的一个领域，其研究成果一直影响着计算机视觉甚至深度学习的发展，图像分类有很多子领域，如多标签分类、细粒度分类等，此处只对单标签图像分类做一个简述。

具体图像分类算法介绍详见[文档](../algorithm_introduction/image_classification.md)。

## 目录

- [1. 数据集介绍](#1)
  - [1.1 ImageNet-1k](#1.1)
  - [1.2 CIFAR-10/CIFAR-100](#1.2)
- [2. 图像分类的流程](#2)
  - [2.1 数据及其预处理](#2.1)
  - [2.2 模型准备](#2.2)
  - [2.3 模型训练](#2.3)
  - [2.4 模型评估](#2.4)
- [3. 使用方法介绍](#3)
  - [3.1 基于 CPU /单卡 GPU 上的训练与评估](#3.1)
      - [3.1.1 模型训练](#3.1.1)
      - [3.1.2 模型微调](#3.1.2)
      - [3.1.3 模型恢复训练](#3.1.3)
      - [3.1.4 模型评估](#3.1.4)
  - [3.2 基于 Linux + 多卡 GPU 的模型训练与评估](#3.2)
      - [3.2.1 模型训练](#3.2.1)
      - [3.2.2 模型微调](#3.2.2)
      - [3.2.3 模型恢复训练](#3.2.3)
      - [3.2.4 模型评估](#3.2.4)
  - [3.3 使用预训练模型进行模型预测](#3.3)
  - [3.4 使用 inference 模型进行模型推理](#3.4)


<a name="1"></a>
## 1. 数据集介绍

<a name="1.1"></a>
### 1.1 ImageNet-1k

ImageNet 项目是一个大型视觉数据库，用于视觉目标识别软件研究。该项目已手动注释了 1400 多万张图像，以指出图片中的对象，并在至少 100 万张图像中提供了边框。ImageNet-1k 是 ImageNet 数据集的子集，其包含 1000 个类别。训练集包含 1281167 个图像数据，验证集包含 50000 个图像数据。2010 年以来，ImageNet 项目每年举办一次图像分类竞赛，即 ImageNet 大规模视觉识别挑战赛(ILSVRC)。挑战赛使用的数据集即为 ImageNet-1k。到目前为止，ImageNet-1k 已经成为计算机视觉领域发展的最重要的数据集之一，其促进了整个计算机视觉的发展，很多计算机视觉下游任务的初始化模型都是基于该数据集训练得到的权重。

<a name="1.2"></a>
### 1.2 CIFAR-10/CIFAR-100

CIFAR-10 数据集由 10 个类的 60000 个彩色图像组成，图像分辨率为 32x32，每个类有 6000 个图像，其中训练集 5000 张，验证集 1000 张，10 个不同的类代表飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、轮船和卡车。CIFAR-100 数据集是 CIFAR-10 的扩展，由 100 个类的 60000 个彩色图像组成，图像分辨率为 32x32，每个类有 600 个图像，其中训练集 500 张，验证集 100 张。由于这两个数据集规模较小，因此可以让研究人员快速尝试不同的算法。这两个数据集也是图像分类领域测试模型好坏的常用数据集。

<a name="2"></a>
## 2. 图像分类的流程

将准备好的训练数据做相应的数据预处理后经过图像分类模型，模型的输出与真实标签做交叉熵损失函数，该损失函数描述了模型的收敛方向，遍历所有的图片数据输入模型，对最终损失函数通过某些优化器做相应的梯度下降，将梯度信息回传到模型中，更新模型的权重，如此循环往复遍历多次数据，即可得到一个图像分类的模型。

<a name="2.1"></a>
### 2.1 数据及其预处理

数据的质量及数量往往可以决定一个模型的好坏。在图像分类领域，数据包括图像及标签。在大部分情形下，带有标签的数据比较匮乏，所以数量很难达到使模型饱和的程度，为了可以使模型学习更多的图像特征，图像数据在进入模型之前要经过很多图像变换或者数据增强，来保证输入图像数据的多样性，从而保证模型有更好的泛化能力。PaddleClas 提供了训练 ImageNet-1k 的标准图像变换，也提供了多种数据增强的方法，相关代码可以查看[数据处理](../../../ppcls/data/preprocess)，配置文件可以参考[数据增强配置文件](../../../ppcls/configs/ImageNet/DataAugment)，相关数据增强算法详见[增强介绍文档](../algorithm_introduction/DataAugmentation.md)。

<a name="2.2"></a>

### 2.2 模型准备

在数据确定后，模型往往决定了最终算法精度的上限，在图像分类领域，经典的模型层出不穷，PaddleClas 提供了 35 个系列共 164 个 ImageNet 预训练模型。具体的精度、速度等指标请参考[骨干网络和预训练模型库](../algorithm_introduction/ImageNet_models.md)。

<a name="2.3"></a>
### 2.3 模型训练

在准备好数据、模型后，便可以开始迭代模型并更新模型的参数。经过多次迭代最终可以得到训练好的模型来做图像分类任务。图像分类的训练过程需要很多经验，涉及很多超参数的设置，PaddleClas 提供了一些列的[训练调优方法](./train_strategy.md)，可以快速助你获得高精度的模型。

同时，PaddleClas 还支持使用VisualDL 可视化训练过程。VisualDL 是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。更多细节请查看[VisualDL](../others/VisualDL.md)。

<a name="2.4"></a>
### 2.4 模型评估

当训练得到一个模型之后，如何确定模型的好坏，需要将模型在验证集上进行评估。评估指标一般是 Top1-Acc 或者 Top5-Acc，该指标越高往往代表模型性能越好。

<a name="3"></a>
## 3. 使用方法介绍

请参考[安装指南](../installation/install_paddleclas.md)配置运行环境，并根据[快速开始](../quick_start/quick_start_classification_new_user.md)文档准备 flower102 数据集，本章节下面所有的实验均以 flower102 数据集为例。

PaddleClas 目前支持的训练/评估环境如下：
```shell
└── CPU/单卡 GPU
    ├── Linux
    └── Windows

└── 多卡 GPU
    └── Linux
```

<a name="3.1"></a>
### 3.1 基于 CPU/单卡 GPU 上的训练与评估

在基于 CPU/单卡 GPU 上训练与评估，推荐使用 `tools/train.py` 与 `tools/eval.py` 脚本。关于 Linux 平台多卡 GPU 环境下的训练与评估，请参考 [3.2. 基于 Linux+GPU 的模型训练与评估](#3.2)。


<a name="3.1.1"></a>
#### 3.1.1 模型训练

准备好配置文件之后，可以使用下面的方式启动训练。

```
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Arch.pretrained=False \
    -o Global.device=gpu
```

其中，`-c` 用于指定配置文件的路径，`-o` 用于指定需要修改或者添加的参数，其中 `-o Arch.pretrained=False` 表示不使用预训练模型，`-o Global.device=gpu` 表示使用 GPU 进行训练。如果希望使用 CPU 进行训练，则需要将 `Global.device` 设置为 `cpu`。

更详细的训练配置，也可以直接修改模型对应的配置文件。具体配置参数参考[配置文档](config_description.md)。

运行上述命令，可以看到输出日志，示例如下：

* 如果在训练中使用了 mixup 或者 cutmix 的数据增广方式，那么日志中将不会打印 top-1 与 top-k（默认为 5）信息：
    ```
    ...
    [Train][Epoch 3/20][Avg]CELoss: 6.46287, loss: 6.46287
    ...
    [Eval][Epoch 3][Avg]CELoss: 5.94309, loss: 5.94309, top1: 0.01961, top5: 0.07941
    ...
    ```

* 如果训练过程中没有使用 mixup 或者 cutmix 的数据增广，那么除了上述信息外，日志中也会打印出 top-1 与 top-k（默认为 5）的信息：

    ```
    ...
    [Train][Epoch 3/20][Avg]CELoss: 6.12570, loss: 6.12570, top1: 0.01765, top5: 0.06961
    ...
    [Eval][Epoch 3][Avg]CELoss: 5.40727, loss: 5.40727, top1: 0.07549, top5: 0.20980
    ...
    ```

训练期间也可以通过 VisualDL 实时观察 loss 变化，详见 [VisualDL](../others/VisualDL.md)。

<a name="3.1.2"></a>
#### 3.1.2 模型微调

根据自己的数据集路径设置好配置文件后，可以通过加载预训练模型的方式进行微调，如下所示。

```
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Arch.pretrained=True \
    -o Global.device=gpu
```

其中 `Arch.pretrained` 设置为 `True` 表示加载 ImageNet 的预训练模型，此外，`Arch.pretrained` 也可以指定具体的模型权重文件的地址，使用时需要换成自己的预训练模型权重文件的路径。

我们也提供了大量基于 `ImageNet-1k` 数据集的预训练模型，模型列表及下载地址详见[模型库概览](../algorithm_introduction/ImageNet_models.md)。


<a name="3.1.3"></a>
#### 3.1.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件，继续训练：

```
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.checkpoints="./output/MobileNetV3_large_x1_0/epoch_5" \
    -o Global.device=gpu
```

其中配置文件不需要做任何修改，只需要在继续训练时设置 `Global.checkpoints` 参数即可，表示加载的断点权重文件路径，使用该参数会同时加载保存的断点权重和学习率、优化器等信息。

**注意**：

* `-o Global.checkpoints` 参数无需包含断点权重文件的后缀名，上述训练命令会在训练过程中生成如下所示的断点权重文件，若想从断点 `5` 继续训练，则 `Global.checkpoints` 参数只需设置为 `"../output/MobileNetV3_large_x1_0/epoch_5"`，PaddleClas 会自动补充后缀名。output 目录下的文件结构如下所示：

    ```shell
    output
    ├── MobileNetV3_large_x1_0
    │   ├── best_model.pdopt
    │   ├── best_model.pdparams
    │   ├── best_model.pdstates
    │   ├── epoch_1.pdopt
    │   ├── epoch_1.pdparams
    │   ├── epoch_1.pdstates
        .
        .
        .
    ```


<a name="3.1.4"></a>
#### 3.1.4 模型评估

可以通过以下命令进行模型评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

上述命令将使用 `./configs/quick_start/MobileNetV3_large_x1_0.yaml` 作为配置文件，对上述训练得到的模型 `./output/MobileNetV3_large_x1_0/best_model` 进行评估。你也可以通过更改配置文件中的参数来设置评估，也可以通过 `-o` 参数更新配置，如上所示。

可配置的部分评估参数说明如下：
* `Arch.name`：模型名称
* `Global.pretrained_model`：待评估的模型预训练模型文件路径

**注意：** 在加载待评估模型时，需要指定模型文件的路径，但无需包含文件后缀名，PaddleClas 会自动补齐 `.pdparams` 的后缀，如 [3.1.3 模型恢复训练](#3.1.3)。


<a name="3.2"></a>
### 3.2 基于 Linux + 多卡 GPU 的模型训练与评估

如果机器环境为 Linux + GPU，那么推荐使用 `paddle.distributed.launch` 启动模型训练脚本(`tools/train.py`)、评估脚本(`tools/eval.py`)，可以更方便地启动多卡训练与评估。

<a name="3.2.1"></a>

#### 3.2.1 模型训练

参考如下方式启动模型训练，`paddle.distributed.launch` 通过设置 `gpus` 指定 GPU 运行卡号：

```bash
# PaddleClas 通过 launch 方式启动多卡多进程训练

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml
```

输出日志信息的格式同上，详见 [3.1.1 模型训练](#3.1.1)。

<a name="3.2.2"></a>
#### 3.2.2 模型微调

根据自己的数据集配置好配置文件之后，可以加载预训练模型进行微调，如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Arch.pretrained=True
```

其中 `Arch.pretrained` 为 `True` 或 `False`，当然也可以设置加载预训练权重文件的路径，使用时需要换成自己的预训练模型权重文件路径，也可以直接在配置文件中修改该路径。

30 分钟玩转 PaddleClas [尝鲜版](../quick_start/quick_start_classification_new_user.md)与[进阶版](../quick_start/quick_start_classification_professional.md)中包含大量模型微调的示例，可以参考该章节在特定的数据集上进行模型微调。

<a name="3.2.3"></a>

#### 3.2.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件继续训练。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Global.checkpoints="./output/MobileNetV3_large_x1_0/epoch_5" \
        -o Global.device=gpu
```

其中配置文件不需要做任何修改，只需要在训练时设置 `Global.checkpoints` 参数即可，该参数表示加载的断点权重文件路径，使用该参数会同时加载保存的模型参数权重和学习率、优化器等信息，详见 [3.1.3 模型恢复训练](#3.1.3)。

<a name="3.2.4"></a>

#### 3.2.4 模型评估

可以通过以下命令进行模型评估。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    tools/eval.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

参数说明详见 [3.1.4 模型评估](#3.1.4)。


<a name="3.3"></a>
### 3.3 使用预训练模型进行模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Infer.infer_imgs=dataset/flowers102/jpg/image_00001.jpg \
    -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

参数说明：
+ `Infer.infer_imgs`：待预测的图片文件路径或者批量预测时的图片文件夹。
+ `Global.pretrained_model`：模型权重文件路径，如 `./output/MobileNetV3_large_x1_0/best_model`


<a name="3.4"></a>
### 3.4 使用 inference 模型进行模型推理

通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.pretrained_model=output/MobileNetV3_large_x1_0/best_model
```


其中，`Global.pretrained_model` 用于指定模型文件路径，该路径仍无需包含模型文件后缀名（如 [3.1.3 模型恢复训练](#3.1.3)）。

上述命令将生成模型结构文件(`inference.pdmodel`)和模型权重文件(`inference.pdiparams`)，然后可以使用预测引擎进行推理：

进入 deploy 目录下：

```bash
cd deploy
```

执行命令进行预测，由于默认 `class_id_map_file` 是 ImageNet 数据集的映射文件，所以此处需要置 None。

```bash
python3 python/predict_cls.py \
    -c configs/inference_cls.yaml \
    -o Global.infer_imgs=../dataset/flowers102/jpg/image_00001.jpg \
    -o Global.inference_model_dir=../inference/ \
    -o PostProcess.Topk.class_id_map_file=None

```
其中：
+ `Global.infer_imgs`：待预测的图片文件路径。
+ `Global.inference_model_dir`：inference 模型结构文件路径，如 `../inference/inference.pdmodel`
+ `Global.use_tensorrt`：是否使用 TesorRT 预测引擎，默认值：`False`
+ `Global.use_gpu`：是否使用 GPU 预测，默认值：`True`
+ `Global.enable_mkldnn`：是否启用 `MKL-DNN` 加速，默认为 `False`。注意 `enable_mkldnn` 与 `use_gpu` 同时为 `True` 时，将忽略 `enable_mkldnn`，而使用 GPU 运行。
+ `Global.use_fp16`：是否启用 `FP16`，默认为 `False`。


注意: 如果使用 `Transformer` 系列模型，如 `DeiT_***_384`, `ViT_***_384` 等，请注意模型的输入数据尺寸，需要设置参数 `resize_short=384`, `resize=384`。

如果你希望提升评测模型速度，使用 GPU 评测时，建议开启 TensorRT 加速预测，使用 CPU 评测时，建议开启 MKLDNN 加速预测。
