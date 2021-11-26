# 图像分类任务数据集说明

本文档将介绍 PaddleClas 所使用的图像分类任务数据集格式，以及图像分类领域的常见数据集介绍。

---

## 目录

- [1. 数据集格式说明](#1)
- [2. 图像分类任务常见数据集介绍](#2)
    - [2.1 ImageNet1k](#2.1)
    - [2.2 Flowers102](#2.2)
    - [2.3 CIFAR10 / CIFAR100](#2.3)
    - [2.4 MNIST](#2.4)
    - [2.5 NUS-WIDE](#2.5)

<a name="1"></a>
## 1. 数据集格式说明

PaddleClas 使用 `txt` 格式文件指定训练集和测试集，以 `ImageNet1k` 数据集为例，其中 `train_list.txt` 和 `val_list.txt` 的格式形如：

```shell
# 每一行采用"空格"分隔图像路径与标注

# 下面是 train_list.txt 中的格式样例
train/n01440764/n01440764_10026.JPEG 0
...

# 下面是 val_list.txt 中的格式样例
val/ILSVRC2012_val_00000001.JPEG 65
...
```
<a name="2"></a>
## 2. 图像分类任务常见数据集介绍

这里整理了常用的图像分类任务数据集，持续更新中，欢迎各位小伙伴补充完善～

<a name="2.1"></a>
### 2.1 ImageNet1k

[ImageNet](https://image-net.org/)项目是一个大型视觉数据库，用于视觉目标识别研究任务，该项目已手动标注了 1400 多万张图像。ImageNet-1k 是 ImageNet 数据集的子集，其包含 1000 个类别。训练集包含 1281167 个图像数据，验证集包含 50000 个图像数据。2010 年以来，ImageNet 项目每年举办一次图像分类竞赛，即 ImageNet 大规模视觉识别挑战赛（ILSVRC）。挑战赛使用的数据集即为 ImageNet-1k。到目前为止，ImageNet-1k 已经成为计算机视觉领域发展的最重要的数据集之一，其促进了整个计算机视觉的发展，很多计算机视觉下游任务的初始化模型都是基于该数据集训练得到的。

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
[ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 |

从官方下载数据后，按如下格式组织数据，即可在 PaddleClas 中使用 ImageNet1k 数据集进行训练。

```bash
PaddleClas/dataset/ILSVRC2012/
|_ train/
|  |_ n01440764
|  |  |_ n01440764_10026.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ n15075141
|     |_ ...
|     |_ n15075141_9993.JPEG
|_ val/
|  |_ ILSVRC2012_val_00000001.JPEG
|  |_ ...
|  |_ ILSVRC2012_val_00050000.JPEG
|_ train_list.txt
|_ val_list.txt
```

<a name="2.2"></a>
### 2.2 Flowers102

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
[flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)|1k | 6k | 102 |

将下载的数据解压后，可以看到以下目录

```shell
jpg/
setid.mat
imagelabels.mat
```

将以上文件放置在 `PaddleClas/dataset/flowers102/` 下

通过运行 `generate_flowers102_list.py` 生成 `train_list.txt` 和 `val_list.txt`：

```shell
python generate_flowers102_list.py jpg train > train_list.txt
python generate_flowers102_list.py jpg valid > val_list.txt
```

按照如下结构组织数据：

```shell
PaddleClas/dataset/flowers102/
|_ jpg/
|  |_ image_03601.jpg
|  |_ ...
|  |_ image_02355.jpg
|_ train_list.txt
|_ val_list.txt
```

<a name="2.3"></a>
### 2.3 CIFAR10 / CIFAR100

CIFAR-10 数据集由 10 个类的 60000 个彩色图像组成，图像分辨率为 32x32，每个类有 6000 个图像，其中训练集 5000 张，验证集 1000 张，10 个不同的类代表飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、轮船和卡车。CIFAR-100 数据集是 CIFAR-10 的扩展，由 100 个类的 60000 个彩色图像组成，图像分辨率为 32x32，每个类有 600 个图像，其中训练集 500 张，验证集 100 张。

数据集地址：http://www.cs.toronto.edu/~kriz/cifar.html

<a name="2.4"></a>
### 2.4 MNIST

MMNIST 是一个非常有名的手写体数字识别数据集，在很多资料中，这个数据集都会被用作深度学习的入门样例。其包含 60000 张图片数据，50000 张作为训练集，10000 张作为验证集，每张图片的大小为 28 * 28。

数据集地址：http://yann.lecun.com/exdb/mnist/

<a name="2.5"></a>
### 2.5 NUS-WIDE

NUS-WIDE 是一个多分类数据集。该数据集包含 269648 张图片, 81 个类别，每张图片被标记为该 81 个类别中的某一类或某几类。

数据集地址：https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
