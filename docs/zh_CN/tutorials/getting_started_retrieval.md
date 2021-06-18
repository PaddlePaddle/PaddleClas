# 开始使用
---
首先，请参考[安装指南](./install.md)配置运行环境。

PaddleClas图像检索部分目前支持的训练/评估环境如下：
```shell
└── CPU/单卡GPU
    ├── Linux
    └── Windows
```
## 目录

* [1. 数据准备与处理](#数据准备与处理)
* [2. 基于单卡GPU上的训练与评估](#基于单卡GPU上的训练与评估)
  * [2.1 模型训练](#模型训练)
  * [2.2 模型恢复训练](#模型恢复训练)
  * [2.3 模型评估](#模型评估)
* [3. 导出inference模型](#导出inference模型)
  
<a name="数据的准备与处理"></a>   
## 1. 数据的准备与处理

* 进入PaddleClas目录。

```bash
## linux or mac， $path_to_PaddleClas表示PaddleClas的根目录，用户需要根据自己的真实目录修改
cd $path_to_PaddleClas
```

* 进入`dataset`目录，为了快速体验PaddleClas图像检索模块，此处使用的数据集为[CUB_200_2011](http://vision.ucsd.edu/sites/default/files/WelinderEtal10_CUB-200.pdf)，其是一个包含200类鸟的细粒度鸟类数据集。首先，下载CUB_200_2011数据集，下载方式请参考[官网](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)。

```shell
# linux or mac
cd dataset

# 将下载后的数据拷贝到此目录
cp {数据存放的路径}/CUB_200_2011.tgz .

# 解压
tar -xzvf CUB_200_2011.tgz

#进入CUB_200_2011目录
cd CUB_200_2011
```

该数据集在用作图像检索任务时，通常将前100类当做训练集，后100类当做测试集，所以此处需要将下载的数据集做一些后处理，来更好的适应PaddleClas的图像检索训练。

```shell
#新建train和test目录
mkdir train && mkdir test

#将数据分成训练集和测试集，前100类作为训练集，后100类作为测试集
ls images | awk -F "." '{if(int($1)<101)print "mv images/"$0" train/"int($1)}' | sh
ls images | awk -F "." '{if(int($1)>100)print "mv images/"$0" test/"int($1)}' | sh

#生成train_list和test_list
tree -r -i -f train | grep jpg | awk -F "/" '{print $0" "int($2) " "NR}' > train_list.txt
tree -r -i -f test | grep jpg | awk -F "/" '{print $0" "int($2) " "NR}' > test_list.txt
```

至此，现在已经得到`CUB_200_2011`的训练集（`train`目录）、测试集（`test`目录）、`train_list.txt`、`test_list.txt`。

数据处理完毕后，`CUB_200_2011`中的`train`目录下应有如下结构：

```
├── 1
│   ├── Black_Footed_Albatross_0001_796111.jpg
│   ├── Black_Footed_Albatross_0002_55.jpg
 ...
├── 10
│   ├── Red_Winged_Blackbird_0001_3695.jpg
│   ├── Red_Winged_Blackbird_0005_5636.jpg
...
```

`train_list.txt`应为：

```
train/99/Ovenbird_0137_92639.jpg 99 1
train/99/Ovenbird_0136_92859.jpg 99 2
train/99/Ovenbird_0135_93168.jpg 99 3
train/99/Ovenbird_0131_92559.jpg 99 4
train/99/Ovenbird_0130_92452.jpg 99 5
...
```
其中，分隔符为空格" ", 三列数据的含义分别是训练数据的路径、训练数据的label信息、训练数据的unique id。

测试集格式与训练集格式相同。

**注意**：

* 当gallery dataset和query dataset相同时，为了去掉检索得到的第一个数据（检索图片本身无须评估），每个数据需要对应一个unique id，用于后续评测mAP、recall@1等指标。关于gallery dataset与query dataset的解析请参考[图像检索数据集介绍](#图像检索数据集介绍), 关于mAP、recall@1等评测指标请参考[图像检索评价指标](#图像检索评价指标)。

返回`PaddleClas`根目录

```shell
# linux or mac
cd ../../
```

<a name="基于单卡GPU上的训练与评估"></a>  
## 2. 基于单卡GPU上的训练与评估

在基于单卡GPU上训练与评估，推荐使用`tools/train.py`与`tools/eval.py`脚本。

<a name="模型训练"></a>
### 2.1 模型训练

准备好配置文件之后，可以使用下面的方式启动图像检索任务的训练。PaddleClas训练图像检索任务的方法是度量学习，关于度量学习的解析请参考[度量学习](#度量学习)。

```
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Arch.Backbone.pretrained=True \
    -o Global.device=gpu
```

其中，`-c`用于指定配置文件的路径，`-o`用于指定需要修改或者添加的参数，其中`-o Arch.Backbone.pretrained=True`表示Backbone部分使用预训练模型，此外，`Arch.Backbone.pretrained`也可以指定具体的模型权重文件的地址，使用时需要换成自己的预训练模型权重文件的路径。`-o Global.device=gpu`表示使用GPU进行训练。如果希望使用CPU进行训练，则需要将`Global.device`设置为`cpu`。

更详细的训练配置，也可以直接修改模型对应的配置文件。具体配置参数参考[配置文档](config.md)。

运行上述命令，可以看到输出日志，示例如下：

    ```
    ...
    [Train][Epoch 1/50][Avg]CELoss: 6.59110, TripletLossV2: 0.54044, loss: 7.13154
    ...
    [Eval][Epoch 1][Avg]recall1: 0.46962, recall5: 0.75608, mAP: 0.21238
    ...
    ```
此处配置文件的Backbone是MobileNetV1，如果想使用其他Backbone，可以重写参数`Arch.Backbone.name`，比如命令中增加`-o Arch.Backbone.name={其他Backbone}`。此外，由于不同模型`Neck`部分的输入维度不同，更换Backbone后可能需要改写此处的输入大小，改写方式类似替换Backbone的名字。

在训练Loss部分，此处使用了[CELoss](../../../ppcls/loss/celoss.py)和[TripletLossV2](../../../ppcls/loss/triplet.py)，配置文件如下：

```
Loss:
  Train:
    - CELoss:
        weight: 1.0
    - TripletLossV2:
        weight: 1.0
        margin: 0.5
```
    
最终的总Loss是所有Loss的加权和，其中weight定义了特定Loss在最终总Loss的权重。如果想替换其他Loss，也可以在配置文件中更改Loss字段，目前支持的Loss请参考[Loss](../../../ppcls/loss)。

<a name="模型恢复训练"></a>
### 2.2 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件，继续训练：

```
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.checkpoints="./output/RecModel/epoch_5" \
    -o Global.device=gpu
```

其中配置文件不需要做任何修改，只需要在继续训练时设置`Global.checkpoints`参数即可，表示加载的断点权重文件路径，使用该参数会同时加载保存的断点权重和学习率、优化器等信息。

**注意**：

* `-o Global.checkpoints`参数无需包含断点权重文件的后缀名，上述训练命令会在训练过程中生成如下所示的断点权重文件，若想从断点`5`继续训练，则`Global.checkpoints`参数只需设置为`"./output/RecModel/epoch_5"`，PaddleClas会自动补充后缀名。

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

<a name="模型评估"></a>
### 2.3 模型评估

可以通过以下命令进行模型评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.pretrained_model=./output/RecModel/best_model
```

上述命令将使用`./configs/quick_start/MobileNetV1_retrieval.yaml`作为配置文件，对上述训练得到的模型`./output/RecModel/best_model`进行评估。你也可以通过更改配置文件中的参数来设置评估，也可以通过`-o`参数更新配置，如上所示。

可配置的部分评估参数说明如下：
* `Arch.name`：模型名称
* `Global.pretrained_model`：待评估的模型的预训练模型文件路径，不同于`Global.Backbone.pretrained`，此处的预训练模型是整个模型的权重，而`Global.Backbone.pretrained`只是Backbone部分的权重。当需要做模型评估时，需要加载整个模型的权重。
* `Metric.Eval`：待评估的指标，默认评估recall@1、recall@5、mAP。当你不准备评测某一项指标时，可以将对应的试标从配置文件中删除；当你想增加某一项评测指标时，也可以参考[Metric](../../../ppcls/metric/metrics.py)部分在配置文件`Metric.Eval`中添加相关的指标。

**注意：** 

* 在加载待评估模型时，需要指定模型文件的路径，但无需包含文件后缀名，PaddleClas会自动补齐`.pdparams`的后缀，如[2.2 模型恢复训练](#模型恢复训练)。

* Metric learning任务一般不评测TopkAcc。

<a name="导出inference模型"></a>
## 3. 导出inference模型

通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.pretrained_model=output/RecModel/best_model \
    -o Global.save_inference_dir=./inference
```

其中，`Global.pretrained_model`用于指定模型文件路径，该路径仍无需包含模型文件后缀名（如[2.2 模型恢复训练](#模型恢复训练)）。当执行后，会在当前目录下生成`./inference`目录，目录下包含`inference.pdiparams`、`inference.pdiparams.info`、`inference.pdmodel`文件。`Global.save_inference_dir`可以指定导出inference模型的路径。此处保存的inference模型在embedding特征层做了截断，即模型最终的输出为n维embedding特征。

上述命令将生成模型结构文件（`inference.pdmodel`）和模型权重文件（`inference.pdiparams`），然后可以使用预测引擎进行推理。使用inference模型推理的流程可以参考[基于Python预测引擎预测推理](@shengyu)。

## 基础知识

图像检索指的是给定一个包含特定实例(例如特定目标、场景、物品等)的查询图像，图像检索旨在从数据库图像中找到包含相同实例的图像。不同于图像分类，图像检索解决的是一个开集问题，训练集中可能不包含被识别的图像的类别。图像检索的整体流程为：首先将图像中表示为一个合适的特征向量，其次，对这些图像的特征向量用欧式距离或余弦距离进行最近邻搜索以找到底库中相似的图像，最后，可以使用一些后处理技术对检索结果进行微调，确定被识别图像的类别等信息。所以，决定一个图像检索算法性能的关键在于图像对应的特征向量的好坏。

<a name="度量学习"></a>
- 度量学习（Metric Learning）

度量学习研究如何在一个特定的任务上学习一个距离函数，使得该距离函数能够帮助基于近邻的算法 (kNN、k-means等) 取得较好的性能。深度度量学习 (Deep Metric Learning )是度量学习的一种方法，它的目标是学习一个从原始特征到低维稠密的向量空间 (嵌入空间，embedding space) 的映射，使得同类对象在嵌入空间上使用常用的距离函数 (欧氏距离、cosine距离等) 计算的距离比较近，而不同类的对象之间的距离则比较远。深度度量学习在计算机视觉领域取得了非常多的成功的应用，比如人脸识别、商品识别、图像检索、行人重识别等。

<a name="图像检索数据集介绍"></a>
- 图像检索数据集介绍

  - 训练集合（train dataset）：用来训练模型，使模型能够学习该集合的图像特征。
  - 底库数据集合（gallery dataset）：用来提供图像检索任务中的底库数据，该集合可与训练集或测试集相同，也可以不同，当与训练集相同时，测试集的类别体系应与训练集的类别体系相同。
  - 测试集合（query dataset）：用来测试模型的好坏，通常要对测试集的每一张测试图片进行特征提取，之后和底库数据的特征进行距离匹配，得到识别结果，后根据识别结果计算整个测试集的指标。

<a name="图像检索评价指标"></a>
- 图像检索评价指标

  <a name="召回率"></a>
  - 召回率（recall）：表示预测为正例且标签为正例的个数 / 标签为正例的个数

    - recall@1：检索的top-1中预测正例且标签为正例的个数 / 标签为正例的个数
    - recall@5：检索的top-5中所有预测正例且标签为正例的个数 / 标签为正例的个数

  <a name="平均检索精度"></a>
  - 平均检索精度(mAP)
  
    - AP: AP指的是不同召回率上的正确率的平均值
    - mAP: 测试集中所有图片对应的AP的的平均值