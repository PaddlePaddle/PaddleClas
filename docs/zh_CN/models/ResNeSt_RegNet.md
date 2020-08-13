# ResNeSt以及RegNet网络

## 概述

ResNeSt系列模型是在2020年提出的，在原有的resnet网络结构上做了改进，通过引入K个Group和在不同Group中加入类似于SEBlock的attention模块，使得精度相比于基础模型ResNet有了大幅度的提高，且参数量和flops与基础的ResNet基本保持一致。

RegNet是由FAIR于2020年提出，旨在深化设计空间理念的概念，在AnyNetX的基础上逐步改进，通过加入共享瓶颈ratio、共享组宽度、调整网络深度与宽度等策略，最终实现简化设计空间结构、提高设计空间的可解释性、改善设计空间的质量，并保持设计空间的模型多样性的目的。最终设计出的模型在类似的条件下，性能还要由于EfficientNet，并且在GPU上的速度提高了5倍。


## 精度、FLOPS和参数量

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeSt50_fast_1s1x64d        | 0.8035 | 0.9528|  0.8035 |            -| 8.68     | 26.3   |
| ResNeSt50        | 0.8102 | 0.9542|  0.8113 |            -| 10.78     | 27.5   |
| RegNetX_4GF        | 0.7850 | 0.9416|  0.7860 |            -| 8.0     | 22.1   |
