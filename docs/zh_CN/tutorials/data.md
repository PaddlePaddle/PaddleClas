# 数据说明

---

## 1.简介
本文档介绍ImageNet1k和Flower102数据准备过程。
以及PaddleClas提供了丰富的[预训练模型](../models/models_intro.md)

## 2.数据集准备

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
[Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)|1k | 6k | 102 | 
[ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 | 

数据格式
按照如下结构组织数据，其中train_list.txt 和val_list.txt的格式形如

```
#path_to_image Class

ILSVRC2012_val_00000001.JPEG 65
...

```
### ImageNet1k
从官方下载数据后，按如下组织数据

```bash
PaddleClas/dataset/imagenet/
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
### Flower
从[VGG官方网站](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)下载后的数据，解压后包括
jpg/
setid.mat
imagelabels.mat
将以上文件放置在PaddleClas/dataset/flower102/下

通过运行generate_flower_list.py生成train_list.txt和val_list.txt

```bash
python generate_flower_list.py jpg train > train_list.txt
python generate_flower_list.py jpg valid > val_list.txt

```
按照如下结构组织数据：

```bash
PaddleClas/dataset/flower102/
|_ jpg/
|  |_ image_03601.jpg 
|  |_ ...
|  |_ image_02355.jpg
|_ train_list.txt
|_ val_list.txt
```


## 3.下载预训练模型
通过tools/download.py下载所需要的预训练模型。

```bash
python tools/download.py -a ResNet50_vd -p ./pretrained -d True
```

参数说明：
+ `architecture`（简写 a）：模型结构
+ `path`（简写 p）：下载路径
+ `decompress` （简写 d）：是否解压
