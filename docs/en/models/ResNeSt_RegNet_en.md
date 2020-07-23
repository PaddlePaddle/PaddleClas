## Overview

The ResNeSt series was proposed in 2020. The original resnet network structure has been improved by introducing K groups and adding an attention module similar to SEBlock in different groups, the accuracy is greater than that of the basic model ResNet, but the parameter amount and flops are almost the same as the basic ResNet.

## Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeSt50        | 0.8102 | 0.9542|  0.8113 |            -|5.39     | 27.5   |
