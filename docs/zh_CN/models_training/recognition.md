# 图像识别
---
在 PaddleClas 中，图像识别，是指给定一张查询图像，系统能够识别该查询图像类别。广义上，图像分类也是图像识别的一种。但是与普通图像识别不同的是，图像分类只能判别出模型已经学习的类别，如果需要添加新的类别，分类模型只能重新训练。PaddleClas 中的图像识别，**对于陌生类别，只需要更新相应的检索库**，就能够正确的识别出查询图像的类别，而无需重新训练模型，这大大增加了识别系统的可用性，同时降低了更新模型的需求，方便用户部署应用。

对于一张待查询图片，PaddleClas 中的图像识别流程主要分为三部分：

1. 主体检测：对于给定一个查询图像，主体检测器首先检测出图像的物体，从而去掉无用背景信息，提高识别精度。
2. 特征提取：对主体检测的各个候选区域，通过特征模型，进行特征提取
3. 特征检索：将提取的特征与特征库中的向量进行相似度比对，得到其标签信息

其中特征库，需要利用已经标注好的图像数据集提前建立。完整的图像识别系统，如下图所示

![](../../images/structure.jpg)
体验整体图像识别系统，或查看特征库建立方法，详见[图像识别快速开始文档](../quick_start/quick_start_recognition.md)。其中，图像识别快速开始文档主要讲解整体流程的使用过程。以下内容，主要对上述三个步骤的训练部分进行介绍。

首先，请参考[安装指南](../installation/install_paddleclas.md)配置运行环境。

## 目录

- [1. 主体检测](#1)
- [2. 特征模型训练](#2)
  - [2.1. 特征模型数据准备与处理](#2.1)
  - [2. 2 特征模型基于单卡 GPU 上的训练与评估](#2.2)
    - [2.2.1 特征模型训练](#2.2.2)
    - [2.2.2 特征模型恢复训练](#2.2.2)
    - [2.2.3 特征模型评估](#2.2.3)
  - [2.3 特征模型导出 inference 模型](#2.3)
- [3. 特征检索](#3)
- [4. 基础知识](#4)

<a name="1"></a>  

## 1. 主体检测

主体检测训练过程基于 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)，唯一的区别在于，主体检测任务中，所有的检测框均属于前景，在这里需要将标注文件中，检测框的 `category_id` 修改为 1，同时将整个标注文件中的 `categories` 映射表修改为下面的格式，即整个类别映射表中只包含 `前景` 类别。

```json
[{u'id': 1, u'name': u'foreground', u'supercategory': u'foreground'}]
```

关于主体检测训练方法可以参考： [PaddleDetection 训练教程](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md#4-%E8%AE%AD%E7%BB%83)。

更多关于 PaddleClas 中提供的主体检测的模型介绍与下载请参考：[主体检测教程](../image_recognition_pipeline/mainbody_detection.md)。

<a name="2"></a>  

## 2. 特征模型训练

<a name="2.1"></a>  

### 2.1 特征模型数据的准备与处理

* 进入 `PaddleClas` 目录。

```bash
## linux or mac， $path_to_PaddleClas 表示 PaddleClas 的根目录，用户需要根据自己的真实目录修改
cd $path_to_PaddleClas
```

* 进入 `dataset` 目录，为了快速体验 PaddleClas 图像检索模块，此处使用的数据集为 [CUB_200_2011](http://vision.ucsd.edu/sites/default/files/WelinderEtal10_CUB-200.pdf)，其是一个包含 200 类鸟的细粒度鸟类数据集。首先，下载 CUB_200_2011 数据集，下载方式请参考[官网](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)。

```shell
# linux or mac
cd dataset

# 将下载后的数据拷贝到此目录
cp {数据存放的路径}/CUB_200_2011.tgz .

# 解压
tar -xzvf CUB_200_2011.tgz

#进入 CUB_200_2011 目录
cd CUB_200_2011
```

该数据集在用作图像检索任务时，通常将前 100 类当做训练集，后 100 类当做测试集，所以此处需要将下载的数据集做一些后处理，来更好的适应 PaddleClas 的图像检索训练。

```shell
#新建 train 和 test 目录
mkdir train && mkdir test

#将数据分成训练集和测试集，前 100 类作为训练集，后 100 类作为测试集
ls images | awk -F "." '{if(int($1)<101)print "mv images/"$0" train/"int($1)}' | sh
ls images | awk -F "." '{if(int($1)>100)print "mv images/"$0" test/"int($1)}' | sh

#生成 train_list 和 test_list
tree -r -i -f train | grep jpg | awk -F "/" '{print $0" "int($2) " "NR}' > train_list.txt
tree -r -i -f test | grep jpg | awk -F "/" '{print $0" "int($2) " "NR}' > test_list.txt
```

至此，现在已经得到 `CUB_200_2011` 的训练集（`train` 目录）、测试集（`test` 目录）、`train_list.txt`、`test_list.txt`。

数据处理完毕后，`CUB_200_2011` 中的 `train` 目录下应有如下结构：

```
├── 1
│   ├── Black_Footed_Albatross_0001_796111.jpg
│   ├── Black_Footed_Albatross_0002_55.jpg
 ...
├── 10
│   ├── Red_Winged_Blackbird_0001_3695.jpg
│   ├── Red_Winged_Blackbird_0005_5636.jpg
...
```

`train_list.txt` 应为：

```
train/99/Ovenbird_0137_92639.jpg 99 1
train/99/Ovenbird_0136_92859.jpg 99 2
train/99/Ovenbird_0135_93168.jpg 99 3
train/99/Ovenbird_0131_92559.jpg 99 4
train/99/Ovenbird_0130_92452.jpg 99 5
...
```
其中，分隔符为空格" ", 三列数据的含义分别是训练数据的路径、训练数据的 label 信息、训练数据的 unique id。

测试集格式与训练集格式相同。

**注意**：

* 当 gallery dataset 和 query dataset 相同时，为了去掉检索得到的第一个数据（检索图片本身无须评估），每个数据需要对应一个 unique id，用于后续评测 mAP、recall@1 等指标。关于 gallery dataset 与 query dataset 的解析请参考[图像检索数据集介绍](#图像检索数据集介绍), 关于 mAP、recall@1 等评测指标请参考[图像检索评价指标](#图像检索评价指标)。

返回 `PaddleClas` 根目录

```shell
# linux or mac
cd ../../
```

<a name="2.2"></a>  

### 2.2 特征模型 GPU 上的训练与评估

在基于单卡 GPU 上训练与评估，推荐使用 `tools/train.py` 与 `tools/eval.py` 脚本。

PaddleClas 支持使用 VisualDL 可视化训练过程。VisualDL 是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。更多细节请查看[VisualDL](../others/VisualDL.md)。

<a name="2.2.1"></a>

#### 2.2.1 特征模型训练

准备好配置文件之后，可以使用下面的方式启动图像检索任务的训练。PaddleClas 训练图像检索任务的方法是度量学习，关于度量学习的解析请参考[度量学习](#度量学习)。

```shell
# 单卡 GPU
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Arch.Backbone.pretrained=True \
    -o Global.device=gpu
# 多卡 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Arch.Backbone.pretrained=True \
    -o Global.device=gpu
```

其中，`-c` 用于指定配置文件的路径，`-o` 用于指定需要修改或者添加的参数，其中 `-o Arch.Backbone.pretrained=True` 表示 Backbone 部分使用预训练模型，此外，`Arch.Backbone.pretrained` 也可以指定具体的模型权重文件的地址，使用时需要换成自己的预训练模型权重文件的路径。`-o Global.device=gpu` 表示使用 GPU 进行训练。如果希望使用 CPU 进行训练，则需要将 `Global.device` 设置为 `cpu`。

更详细的训练配置，也可以直接修改模型对应的配置文件。具体配置参数参考[配置文档](config_description.md)。

运行上述命令，可以看到输出日志，示例如下：

    ```
    ...
    [Train][Epoch 1/50][Avg]CELoss: 6.59110, TripletLossV2: 0.54044, loss: 7.13154
    ...
    [Eval][Epoch 1][Avg]recall1: 0.46962, recall5: 0.75608, mAP: 0.21238
    ...
    ```
此处配置文件的 Backbone 是 MobileNetV1，如果想使用其他 Backbone，可以重写参数 `Arch.Backbone.name`，比如命令中增加 `-o Arch.Backbone.name={其他 Backbone}`。此外，由于不同模型 `Neck` 部分的输入维度不同，更换 Backbone 后可能需要改写此处的输入大小，改写方式类似替换 Backbone 的名字。

在训练 Loss 部分，此处使用了 [CELoss](../../../ppcls/loss/celoss.py) 和 [TripletLossV2](../../../ppcls/loss/triplet.py)，配置文件如下：

```
Loss:
  Train:
    - CELoss:
        weight: 1.0
    - TripletLossV2:
        weight: 1.0
        margin: 0.5
```

最终的总 Loss 是所有 Loss 的加权和，其中 weight 定义了特定 Loss 在最终总 Loss 的权重。如果想替换其他 Loss，也可以在配置文件中更改 Loss 字段，目前支持的 Loss 请参考 [Loss](../../../ppcls/loss)。

<a name="2.2.2"></a>

#### 2.2.2 特征模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件，继续训练：

```shell
# 单卡
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.checkpoints="./output/RecModel/epoch_5" \
    -o Global.device=gpu
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.checkpoints="./output/RecModel/epoch_5" \
    -o Global.device=gpu
```

其中配置文件不需要做任何修改，只需要在继续训练时设置 `Global.checkpoints` 参数即可，表示加载的断点权重文件路径，使用该参数会同时加载保存的断点权重和学习率、优化器等信息。

**注意**：

* `-o Global.checkpoints` 参数无需包含断点权重文件的后缀名，上述训练命令会在训练过程中生成如下所示的断点权重文件，若想从断点 `5` 继续训练，则 `Global.checkpoints` 参数只需设置为 `"./output/RecModel/epoch_5"`，PaddleClas 会自动补充后缀名。

    ```shell
    output/
    └── RecModel
        ├── best_model.pdopt
        ├── best_model.pdparams
        ├── best_model.pdstates
        ├── epoch_1.pdopt
        ├── epoch_1.pdparams
        ├── epoch_1.pdstates
        .
        .
        .
    ```

<a name="2.2.3"></a>

#### 2.2.3 特征模型评估

可以通过以下命令进行模型评估。

```bash
# 单卡
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.pretrained_model=./output/RecModel/best_model
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch tools/eval.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.pretrained_model=./output/RecModel/best_model
```

上述命令将使用 `./configs/quick_start/MobileNetV1_retrieval.yaml` 作为配置文件，对上述训练得到的模型 `./output/RecModel/best_model` 进行评估。你也可以通过更改配置文件中的参数来设置评估，也可以通过 `-o` 参数更新配置，如上所示。

可配置的部分评估参数说明如下：
* `Arch.name`：模型名称
* `Global.pretrained_model`：待评估的模型的预训练模型文件路径，不同于 `Global.Backbone.pretrained`，此处的预训练模型是整个模型的权重，而 `Global.Backbone.pretrained` 只是 Backbone 部分的权重。当需要做模型评估时，需要加载整个模型的权重。
* `Metric.Eval`：待评估的指标，默认评估 recall@1、recall@5、mAP。当你不准备评测某一项指标时，可以将对应的试标从配置文件中删除；当你想增加某一项评测指标时，也可以参考 [Metric](../../../ppcls/metric/metrics.py) 部分在配置文件 `Metric.Eval` 中添加相关的指标。

**注意：**

* 在加载待评估模型时，需要指定模型文件的路径，但无需包含文件后缀名，PaddleClas 会自动补齐 `.pdparams` 的后缀，如 [2.2.2 特征模型恢复训练](#2.2.2)。

* Metric learning 任务一般不评测 TopkAcc。

<a name="2.3"></a>

### 2.3 特征模型导出 inference 模型

通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.pretrained_model=output/RecModel/best_model \
    -o Global.save_inference_dir=./inference
```

其中，`Global.pretrained_model` 用于指定模型文件路径，该路径仍无需包含模型文件后缀名（如[2.2.2 特征模型恢复训练](#2.2.2)）。当执行后，会在当前目录下生成 `./inference` 目录，目录下包含 `inference.pdiparams`、`inference.pdiparams.info`、`inference.pdmodel` 文件。`Global.save_inference_dir` 可以指定导出 inference 模型的路径。此处保存的 inference 模型在 embedding 特征层做了截断，即模型最终的输出为 n 维 embedding 特征。

上述命令将生成模型结构文件(`inference.pdmodel`)和模型权重文件(`inference.pdiparams`)，然后可以使用预测引擎进行推理。使用 inference 模型推理的流程可以参考[基于 Python 预测引擎预测推理](../inference_deployment/python_deploy.md)。

<a name="3"></a>  

## 3. 特征检索

PaddleClas 图像检索部分目前支持的环境如下：

```shell
└── CPU/单卡 GPU
    ├── Linux
    ├── MacOS
    └── Windows
```

此部分使用了 [Faiss](https://github.com/facebookresearch/faiss) 作为检索库，其是一个高效的特征检索及聚类的库。此库中集成了多种相似度检索算法，以满足不同的检索场景。在 PaddleClas 中，支持三种检索算法：

- **HNSW32**: 一种图索引方法。检索精度较高，速度较快。但是特征库只支持添加图像功能，不支持删除图像特征功能。（默认方法）
- **IVF**：倒排索引检索方法。速度较快，但是精度略低。特征库支持增加、删除图像特功能。
- **FLAT**： 暴力检索算法。精度最高，但是数据量大时，检索速度较慢。特征库支持增加、删除图像特征功能。

详细介绍请参考 [Faiss](https://github.com/facebookresearch/faiss) 官方文档。

具体安装方法如下：

```python
pip install faiss-cpu==1.7.1post2
```

若使用时，不能正常引用，则 `uninstall` 之后，重新 `install`，尤其是 `windows` 下。

<a name="4"></a>

## 4. 基础知识

图像检索指的是给定一个包含特定实例(例如特定目标、场景、物品等)的查询图像，图像检索旨在从数据库图像中找到包含相同实例的图像。不同于图像分类，图像检索解决的是一个开集问题，训练集中可能不包含被识别的图像的类别。图像检索的整体流程为：首先将图像中表示为一个合适的特征向量，其次，对这些图像的特征向量用欧式距离或余弦距离进行最近邻搜索以找到底库中相似的图像，最后，可以使用一些后处理技术对检索结果进行微调，确定被识别图像的类别等信息。所以，决定一个图像检索算法性能的关键在于图像对应的特征向量的好坏。

<a name="度量学习"></a>
- 度量学习（Metric Learning）

度量学习研究如何在一个特定的任务上学习一个距离函数，使得该距离函数能够帮助基于近邻的算法（kNN、k-means 等）取得较好的性能。深度度量学习(Deep Metric Learning)是度量学习的一种方法，它的目标是学习一个从原始特征到低维稠密的向量空间（嵌入空间，embedding space）的映射，使得同类对象在嵌入空间上使用常用的距离函数（欧氏距离、cosine 距离等）计算的距离比较近，而不同类的对象之间的距离则比较远。深度度量学习在计算机视觉领域取得了非常多的成功的应用，比如人脸识别、商品识别、图像检索、行人重识别等。更详细的介绍请参考[此文档](../algorithm_introduction/metric_learning.md)。

<a name="图像检索数据集介绍"></a>

- 图像检索数据集介绍

  - 训练集合(train dataset)：用来训练模型，使模型能够学习该集合的图像特征。
  - 底库数据集合(gallery dataset)：用来提供图像检索任务中的底库数据，该集合可与训练集或测试集相同，也可以不同，当与训练集相同时，测试集的类别体系应与训练集的类别体系相同。
  - 测试集合(query dataset)：用来测试模型的好坏，通常要对测试集的每一张测试图片进行特征提取，之后和底库数据的特征进行距离匹配，得到识别结果，后根据识别结果计算整个测试集的指标。

<a name="图像检索评价指标"></a>
- 图像检索评价指标

  <a name="召回率"></a>
  - 召回率(recall)：表示预测为正例且标签为正例的个数 / 标签为正例的个数

    - recall@1：检索的 top-1 中预测正例且标签为正例的个数 / 标签为正例的个数
    - recall@5：检索的 top-5 中所有预测正例且标签为正例的个数 / 标签为正例的个数

  <a name="平均检索精度"></a>
  - 平均检索精度(mAP)

    - AP: AP 指的是不同召回率上的正确率的平均值
    - mAP: 测试集中所有图片对应的 AP 的平均值
