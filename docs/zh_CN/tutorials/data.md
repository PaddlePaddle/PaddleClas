# 数据说明

---

## 1.简介
本文档介绍ImageNet1k和flowers102数据准备过程。

## 2.数据集准备

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
[flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)|1k | 6k | 102 |
[ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 |

* 数据格式
按照如下结构组织数据，其中train_list.txt 和val_list.txt的格式形如

```shell
# 每一行采用"空格"分隔图像路径与标注

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
### Flowers102
从[VGG官方网站](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)下载后的数据，解压后包括

```shell
jpg/
setid.mat
imagelabels.mat
```

将以上文件放置在PaddleClas/dataset/flowers102/下

通过运行generate_flowers102_list.py生成train_list.txt和val_list.txt

```bash
python generate_flowers102_list.py jpg train > train_list.txt
python generate_flowers102_list.py jpg valid > val_list.txt

```
按照如下结构组织数据：

```bash
PaddleClas/dataset/flowers102/
|_ jpg/
|  |_ image_03601.jpg
|  |_ ...
|  |_ image_02355.jpg
|_ train_list.txt
|_ val_list.txt
```
