# ResNeSt以及RegNet网络

## 概述

ResNeSt系列模型是在2020年提出的，在原有的resnet网络结构上做了改进，通过引入K个Group和在不同Group中加入类似于SEBlock的attention模块，使得精度相比于基础模型ResNet有了大幅度的提高，且参数量和flops与基础的ResNet基本保持一致。


## 精度、FLOPS和参数量

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeSt50        | 0.8102 | 0.9542|  0.8113 |            -|5.39     | 27.5   |


