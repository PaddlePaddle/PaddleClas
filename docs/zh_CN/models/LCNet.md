# PPLCNet系列

## 概述

PPLCNet系列是百度PaddleCV团队提出的一种在Intel-CPU上表现优异的网络，作者总结了一些在Intel-CPU上可以提升模型精度但几乎不增加推理耗时的方法，将这些方法组合成了一个新的网络，即PPLCNet。与其他轻量级网络相比，PPLCNet可以在相同延时下取得更高的精度。PPLCNet已在图像分类、目标检测、语义分割上表现出了强大的竞争力。



## 精度、FLOPS和参数量

| Models           | Top1 | Top5 | FLOPs<br>(M) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|
| PPLCNet_x0_25        |0.5186           | 0.7565           | 18    | 1.5  |
| PPLCNet_x0_35        |0.5809           | 0.8083           | 29    | 1.6  |
| PPLCNet_x0_5         |0.6314           | 0.8466           | 47    | 1.9  |
| PPLCNet_x0_75        |0.6818           | 0.8830           | 99    | 2.4  |
| PPLCNet_x1_0         |0.7132           | 0.9003           | 161   | 3.0  |
| PPLCNet_x1_5         |0.7371           | 0.9153           | 342   | 4.5  |
| PPLCNet_x2_0         |0.7518           | 0.9227           | 590   | 6.5  |
| PPLCNet_x2_5         |0.7660           | 0.9300           | 906   | 9.0  |
| PPLCNet_x0_5_ssld    |0.6610           | 0.8646           | 47    | 1.9  |
| PPLCNet_x1_0_ssld    |0.7439           | 0.9209           | 161   | 3.0  |
| PPLCNet_x2_5_ssld    |0.8082           | 0.9533           | 906   | 9.0  |



## 基于Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz的预测速度

| Models                 | Crop Size | Resize Short Size | FP32<br>Batch Size=1<br>(ms) |
|------------------|-----------|-------------------|--------------------------|
| PPLCNet_x0_25        | 224       | 256               | 1.74                    |
| PPLCNet_x0_35        | 224       | 256               | 1.92                    |
| PPLCNet_x0_5         | 224       | 256               | 2.05                    |
| PPLCNet_x0_75        | 224       | 256               | 2.29                    |
| PPLCNet_x1_0         | 224       | 256               | 2.46                    |
| PPLCNet_x1_5         | 224       | 256               | 3.19                    |
| PPLCNet_x2_0         | 224       | 256               | 4.27                    |
| PPLCNet_x2_5         | 224       | 256               | 5.39                    |
| PPLCNet_x0_5_ssld    | 224       | 256               | 2.05                    |
| PPLCNet_x1_0_ssld    | 224       | 256               | 2.46                    |
| PPLCNet_x2_5_ssld    | 224       | 256               | 5.39                    |
