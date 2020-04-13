# 安装说明

---

## 1.简介

本章将介绍如何安装PaddleClas及其依赖项，准备ImageNet1k图像分类数据集和下载预训练模型。


## 2.安装PaddlePaddle

运行PaddleClas需要PaddlePaddle Fluid v1.7或更高版本。请按照[安装文档](http://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle.fluid as fluid
>>> fluid.install_check.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"

```
注意：
- 从源码编译的PaddlePaddle版本号为0.0.0，请确保使用了Fluid v1.7之后的源码编译。
- PaddleClas基于PaddlePaddle高性能的分布式训练能力，若您从源码编译，请确保打开编译选项，**WITH_DISTRIBUTE=ON**。


**环境需求:**

- Python2（官方已不提供更新维护）或Python3 (windows系统仅支持Python3)
- CUDA >= 8.0
- cuDNN >= 5.0
- nccl >= 2.1.2


## 3.安装PaddleClas

**克隆PaddleClas模型库：**

```
cd path_to_clone_PaddleClas
git clone https://github.com/PaddlePaddle/PaddleClas.git
```

**安装Python依赖库：**

Python依赖库在[requirements.txt](https://github.com/PaddlePaddle/PaddleClas/blob/master/requirements.txt)中给出，可通过如下命令安装：

```
pip install --upgrade -r requirements.txt
```


## 4.下载ImageNet1K图像分类数据集

PaddleClas默认支持ImageNet1000分类任务。
在Linux系统下通过如下方式进行数据准备：

```
cd dataset/ILSVRC2012/
sh download_imagenet2012.sh
```
在```download_imagenet2012.sh```脚本中，通过下面三步来准备数据：

**步骤一：** 首先在```image-net.org```网站上完成注册，用于获得一对```Username```和```AccessKey```。

**步骤二：** 从ImageNet官网下载ImageNet-2012的图像数据。训练以及验证数据集会分别被下载到"train" 和 "val" 目录中。注意，ImageNet数据的大小超过140GB，下载非常耗时；已经自行下载ImageNet的用户可以直接将数据按"train" 和 "val" 目录放到```dataset/ILSVRC2012```。

**步骤三：** 下载训练与验证集合对应的标签文件。

* train_list.txt: ImageNet-2012训练集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
train/n02483708/n02483708_2436.jpeg 369
```
* val_list.txt: ImageNet-2012验证集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
val/ILSVRC2012_val_00000001.jpeg 65
```

**Windows系统下请用户自行下载ImageNet数据，[label下载链接](http://paddle-imagenet-models.bj.bcebos.com/ImageNet_label.tgz)**



## 5.下载预训练模型
PaddleClas 提供了丰富的预训练模型，支持的模型列表请参考[模型库](../models/models_intro.md)。
通过tools/download.py可以下载所需要的预训练模型。

```bash
python tools/download.py -a ResNet50_vd -p ./pretrained -d True
```
