# 图像分类任务数据集说明

本文档将介绍 PaddleClas 所使用的数据集格式，以及图像分类任务的主要数据集，包括 ImageNet1k 和 flowers102 的介绍。

---

## 1. 数据集格式说明

PaddleClas 使用 `txt` 格式文件指定训练集和测试集，以 `ImageNet1k` 数据集为例，其中 `train_list.txt` 和 `val_list.txt` 的格式形如：

```shell
# 每一行采用"空格"分隔图像路径与标注

# 下面是train_list.txt中的格式样例
train/n01440764/n01440764_10026.JPEG 0
...

# 下面是val_list.txt中的格式样例
val/ILSVRC2012_val_00000001.JPEG 65
...
```

## 2. ImageNet1k 数据集

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
[ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 |

从官方下载数据后，按如下组织数据

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

## 3. Flowers102 数据集

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
