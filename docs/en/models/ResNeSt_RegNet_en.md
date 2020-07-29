## Overview

The ResNeSt series was proposed in 2020. The original resnet network structure has been improved by introducing K groups and adding an attention module similar to SEBlock in different groups, the accuracy is greater than that of the basic model ResNet, but the parameter amount and flops are almost the same as the basic ResNet.

## Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeSt50_fast_1s1x64d        | 0.8035 | 0.9528|  0.8035 |            -| 8.68     | 26.3   |
| ResNeSt50        | 0.8102 | 0.9542|  0.8113 |            -| 10.78     | 27.5   |
