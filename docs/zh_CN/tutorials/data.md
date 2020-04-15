# 数据说明

---

## 1.简介
PaddleClas支持ImageNet1000和Flower数据分类任务。
PaddleClas提供了丰富的预训练模型，支持的模型列表请参考[模型库](../models/models_intro.md)

## 2.数据集准备

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
Flowers|1k | 6k | 102 | 
[ImageNet](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 | 

数据格式

PaddleClas加载PaddleClas/dataset/中的数据，请将下载后的数据按下面格式组织放置到PaddleClas/dataset/中。

```bash
PaddleClas/dataset/imagenet
|_ train
|  |_ n01440764
|  |  |_ n01440764_10026.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ n15075141
|     |_ ...
|     |_ n15075141_9993.JPEG
|_ val
|  |_ ILSVRC2012_val_00000001.JPEG
|  |_ ...
|  |_ ILSVRC2012_val_00050000.JPEG
|_ train_list.txt
|_ val_list.txt

```bash
PaddleClas/dataset/flower
|_ train
|  |_ image_03601.jpg 
|  |_ ...
|  |_ image_07073.jpg
|_ val
|  |_ image_04121.jpg
|  |_ ...
|  |_ image_02355.jpg
|_ train_list.txt
|_ val_list.txt
```


或是通过软链接将数据从实际地址链接到PaddleClas/dataset/下

```bash
#imagenet
ln -s actual_path/imagenet path_to_PaddleClas/dataset/imagenet

#flower
ln -s actual_path/flower path_to_PaddleClas/dataset/flower

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
