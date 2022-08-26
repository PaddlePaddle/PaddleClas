# PP-ShiTu应用场景介绍

该文档介绍了PP-ShiTu提供的各种应用场景库简介、下载链接以及使用简介。

------

## 目录

- [1. 应用场景介绍](#1-应用场景介绍)
- [2. 使用说明](#2-使用说明)
  - [2.1 下载、解压场景库数据](#21-下载解压场景库数据)
  - [2.2 准备识别模型](#22-准备识别模型)

<a name="1. 应用场景介绍"></a>

## 1. 应用场景介绍

PP-ShiTu应用场景介绍和下载地址如下表所示。

| 场景 |场景简介|Recall@1|场景库下载地址|
|:---:|:---:|:---:|:---:|
| 球类 | 各种球类识别 | --- | --- |
| 球类 | 各种球类识别 | --- | --- |
| 球类 | 各种球类识别 | --- | --- |



<a name="2. 使用说明"></a>

## 2. 使用说明

<a name="2.1 下载、解压场景库数据"></a>

### 2.1 下载、解压场景库数据
首先创建存放场景库的地址`deploy/datasets`，并根据需要选择场景下载对应场景库。

```shell
cd deploy
mkdir datasets
```
将对应场景库解压到`deploy/datasets`中。
```shell
cd datasets
tar xf ***.tar
```
以`dataset_name`为例，解压完毕后，`datasets/dataset_name`文件夹下应有如下文件结构：
```shel
├── dataset_name/
│   ├── gallery/
│   ├── index/
│   ├── query/
├── ...
```
其中，`gallery`文件夹中存放的是用于构建索引库的原始图像，`index`表示基于原始图像构建得到的索引库信息，`query`文件夹存放的是用于检索的图像列表。

<a name="2.2 准备识别模型"></a>

### 2.2 准备识别模型
创建存放模型的文件夹`deploy/models`，并下载轻量级主体检测、识别模型，命令如下：
```shellc
cd ..
mkdir models
cd models

# 下载通用检测 inference 模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar && tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
# 下载识别 inference 模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar && tar -xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
```
