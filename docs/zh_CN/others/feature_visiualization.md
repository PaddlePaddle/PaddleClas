# 特征图可视化指南
-----
## 目录

* [1. 概述](#1)
* [2. 准备工作](#2)
* [3. 修改模型](#3)
* [4. 结果](#4)

<a name='1'></a>

## 1. 概述

特征图是输入图片在卷积网络中的特征表达，对特征图的研究可以有利于我们对于模型的理解与设计，所以基于动态图我们使用本工具来可视化特征图。

<a name='2'></a>

## 2. 准备工作

首先需要选定研究的模型，本文设定 ResNet50 作为研究模型，将模型组网代码[resnet.py](../../../ppcls/arch/backbone/legendary_models/resnet.py)拷贝到[目录](../../../ppcls/utils/feature_maps_visualization/)下，并下载[ResNet50 预训练模型](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams)，或使用以下命令下载。

```bash
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams
```

其他模型网络结构代码及预训练模型请自行下载：[模型库](../../../ppcls/arch/backbone/)，[预训练模型](../algorithm_introduction/ImageNet_models.md)。

 <a name='3'></a>

## 3. 修改模型

找到我们所需要的特征图位置，设置 self.fm 将其 fetch 出来，本文以 resnet50 中的 stem 层之后的特征图为例。

在 ResNet50 的 forward 函数中指定要可视化的特征图

```python
    def forward(self, x):
        with paddle.static.amp.fp16_guard():
            if self.data_format == "NHWC":
                x = paddle.transpose(x, [0, 2, 3, 1])
                x.stop_gradient = True
            x = self.stem(x)
            fm = x
            x = self.max_pool(x)
            x = self.blocks(x)
            x = self.avg_pool(x)
            x = self.flatten(x)
            x = self.fc(x)
        return x, fm
```

然后修改代码[fm_vis.py](../../../ppcls/utils/feature_maps_visualization/fm_vis.py)，引入 `ResNet50`，实例化 `net` 对象：

```python
from resnet import ResNet50
net = ResNet50()
```

最后执行函数

```bash
python tools/feature_maps_visualization/fm_vis.py \
    -i the image you want to test \
    -c channel_num -p pretrained model \
    --show whether to show \
    --interpolation interpolation method\
    --save_path where to save \
    --use_gpu whether to use gpu
```

参数说明：
+ `-i`：待预测的图片文件路径，如 `./test.jpeg`
+ `-c`：特征图维度，如 `5`
+ `-p`：权重文件路径，如 `./ResNet50_pretrained`
+ `--interpolation`: 图像插值方式，默认值 1
+ `--save_path`：保存路径，如：`./tools/`
+ `--use_gpu`：是否使用 GPU 预测，默认值：True

<a name='4'></a>

## 4. 结果

* 输入图片：  

![](../../images/feature_maps/feature_visualization_input.jpg)

* 运行下面的特征图可视化脚本

```
python tools/feature_maps_visualization/fm_vis.py \
    -i ./docs/images/feature_maps/feature_visualization_input.jpg \
    -c 5 \
    -p pretrained/ResNet50_pretrained/  \
    --show=True \
    --interpolation=1 \
    --save_path="./output.png" \
    --use_gpu=False
```

* 输出特征图保存为 `output.png`，如下所示。

![](../../images/feature_maps/feature_visualization_output.jpg)
