# RepVGG系列

## 概述

RepVGG（Making VGG-style ConvNets Great Again）系列模型是由清华大学(丁贵广团队)、旷视科技(孙剑等人)、港科大和阿伯里斯特威斯大学在2021年提出的一个简单但强大的卷积神经网络架构，该架构具有类似于VGG的推理时间主体，该主体仅由3x3卷积和ReLU的堆栈组成，而训练时间模型具有多分支拓扑。训练时间和推理时间架构的这种解耦是通过结构重新参数化(re-parameterization)技术实现的，因此该模型称为RepVGG。[论文地址](https://arxiv.org/abs/2101.03697)。


## 精度、FLOPS和参数量

| Models | Top1 | Top5 | Reference<br>top1| FLOPS<br>(G) |
|:--:|:--:|:--:|:--:|:--:|
| RepVGG_A0 | 0.7131 | 0.9016 | 0.7241 |     |
| RepVGG_A1 | 0.7380 | 0.9146 | 0.7446 |     |
| RepVGG_A2 | 0.7571 | 0.9264 | 0.7648 |     |
| RepVGG_B0 | 0.7450 | 0.9213 | 0.7514 |     |
| RepVGG_B1 | 0.7773 | 0.9385 | 0.7837 |     |
| RepVGG_B2 | 0.7813 | 0.9410 | 0.7878 |     |
| RepVGG_B1g2 | 0.7732 | 0.9359 | 0.7778 |    |
| RepVGG_B1g4 | 0.7675 | 0.9335 | 0.7758 |    |
| RepVGG_B2g4 | 0.7881 | 0.9448 | 0.7938 |    |
| RepVGG_B3g4 | 0.7965 | 0.9485 | 0.8021 |    |

关于Params、FLOPs、Inference speed等信息，敬请期待。
