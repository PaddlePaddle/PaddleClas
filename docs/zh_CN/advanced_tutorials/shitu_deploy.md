# PP-ShiTu迁移应用及调优

## 目录

- [1.使用官方模型](#1)
  - [1.1 下载官方模型及数据准备](#1.1)
  - [1.2 建立检索库（gallery）](#1.2)
  - [1.3 精度测试](#1.3)
- [2.模型调优](#2)
  - [2.1 检测模型调优](#2.1)
  - [2.2 识别模型调优](#2.2)
- [3.模型加速](#3)

在[PP-ShiTu快速开始](../quick_start/quick_start_recognition.md)中，主要展示了`PP-ShiTu`的快速开始demo。那么本文档主要介绍，如何将`PP-ShiTu`应用到自己的需求中，及如何继续调优，优化识别效果。具体可以分成以下三种情况

- 直接使用官方模型
- 根据需求对模型进行调优
- 模型加速

以下分别对上面三种情况进行具体讲解。

<a name="1"></a>

## 1. 使用官方模型

对于其他的应用，首先建议使用此种方法，因为PaddleClas官方模型，是经过大量数据训练得到的，具有强大泛化性能的轻量级通用模型。其速度较快、精度较高，在Intel cpu上直接部署，也能快速得到结果。同时，直接使用官方模型，则无需训练新的模型，节约时间，方便快速部署。

使用官方模型具体步骤如下：

- 下载模型及数据准备
- 检索库更新
- 精度测试

<a name="1.1"></a>

### 1.1 下载官方模型及数据准备

模型下载及pipline 运行详见[图像识别快速开始](../quick_start/quick_start_recognition.md)

下载模型后，要准备相应的数据，即所迁移应用的具体数据，数据量根据实际情况，自行决定，但是不能太少，会影响精度。将准备的数据分成两部分：1）建库图像（gallery），2）测试图像。其中建库数据无需过多，但需保证每个类别包含此类别物体不同角度的图像，建议每个类别至少5张图，请根据实际情况，具体调节。

数据标注工具可以使用[lebalme](https://github.com/wkentaro/labelme)。标注数据时。请标注待识别物体的的包围框（BoundingBox），注意只需要标注**建库图像**。。

建议一个类别一共准备30张图左右，其中约至少5张图作为建库图像，剩下的作为测试图像。

<a name="1.2"></a>

### 1.2 建立检索库（gallery）

对于加入检索的数据，每个类别尽量准备此类别的各角度的图像，丰富类别信息。准备的图像只能包含此类别，同时图像背景尽可能的少、简单。即将要加入检索根据标注的包围框信息，裁剪出bbox图像作为新的要加入的图像，以提高检索库的图像质量。

收集好图像后，数据整理及建库流程详见[图像识别快速开始](../quick_start/quick_start_recognition.md)中`3.2 建立新的索引库`

<a name="1.3"></a>

### 1.3 精度测试

使用测试图像，对整个pipline进行简单的精度测试。如发现类别不正确，则需对gallery进行调整，将不正确的测试图像的相似图片（标注并裁剪出没有背景的物体）加入gallery中，反复迭代。经过调整后，可以测试出整个pipeline的精度。如果精度能够满足需求，则可继续使用。若精度不达预期，则需对模型进行调优，参考下面文档。

<a name="2"></a>

## 2. 模型调优

在使用官方模型之后，如果发现精度不达预期，则可对模型进行训练调优。同时，根据官方模型的结果，需要进一步大概判断出 检测模型精度、还是识别模型精度问题。不同模型的调优，可参考以下文档。

<a name="2.1"></a>

### 2.1 检测模型调优

`PP-ShiTu`中检测模型采用的 `PicoDet    `算法，具体算法请参考[此文档](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet)。检测模型的训练及调优，请参考[此文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/README_cn.md)。

对模型进行训练的话，需要自行准备数据，并对数据进行标注，建议一个类别至少准备200张标注图像，并将标注图像及groudtruth文件转成coco文件格式，以方便使用PaddleDetection进行训练。主体检测的预训练权重及相关配置文件相见[主体检测文档](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/application/mainbody_detection)。训练的时候，请加载主体检测的预训练权重。

<a name="2.2"></a>

### 2.2 识别模型调优

在使用官方模型后，如果不满足精度需求，则可以参考此部分文档，进行模型调优

因为要对模型进行训练，所以收集自己的数据集。数据准备及相应格式请参考：[特征提取文档](../image_recognition_pipeline/feature_extraction.md)中 `4.1数据准备`部分、[识别数据集说明](../data_preparation/recognition_dataset.md)。值得注意的是，此部分需要准备大量的数据，以保证识别模型效果。训练配置文件参考：[通用识别模型配置文件](../../../ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5.yaml)，训练方法参考：[识别模型训练](../models_training/recognition.md)

- 数据增强：根据实际情况选择不同数据增强方法。如：实际应用中数据遮挡比较严重，建议添加`RandomErasing`增强方法。详见[数据增强文档](./DataAugmentation.md)
- 换不同的`backbone`，一般来说，越大的模型，特征提取能力更强。不同`backbone`详见[模型介绍](../algorithm_introduction/ImageNet_models.md)
- 选择不同的`Metric Learning`方法。不同的`Metric Learning`方法，对不同的数据集效果可能不太一样，建议尝试其他`Loss`,详见[Metric Learning](../algorithm_introduction/metric_learning.md)
- 采用蒸馏方法，对小模型进行模型能力提升，详见[模型蒸馏](../algorithm_introduction/knowledge_distillation.md)
- 增补数据集。针对错误样本，添加badcase数据

模型训练完成后，参照[1.2 检索库更新](#1.2)进行检索库更新。同时，对整个pipeline进行测试，如果精度不达预期，则重复此步骤。

<a name="3"></a>

## 3. 模型加速

模型加速主要以下几种方法：

- 替换小模型：一般来说，越小的模型预测速度相对越快
- 模型裁剪、量化：请参考文档[模型压缩](./model_prune_quantization.md)，压缩配置文件修改请参考[slim相关配置文件](../../../ppcls/configs/slim/)。
