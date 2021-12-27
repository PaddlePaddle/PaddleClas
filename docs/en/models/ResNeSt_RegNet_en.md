# ResNeSt and RegNet series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)
* [3. Inference speed based on T4 GPU](#3)

<a name='1'></a>
## 1. Overview

The ResNeSt series was proposed in 2020. The original resnet network structure has been improved by introducing K groups and adding an attention module similar to SEBlock in different groups, the accuracy is greater than that of the basic model ResNet, but the parameter amount and flops are almost the same as the basic ResNet.

RegNet was proposed in 2020 by Facebook to deepen the concept of design space. Based on AnyNetX, the model performance is gradually improved by shared bottleneck ratio, shared group width, adjusting network depth or width and other strategies. What's more, the design space structure is simplified, whose interpretability is also be improved. The quality of design space is improved while its diversity is maintained. Under similar conditions, the performance of the designed RegNet model performs better than EfficientNet and 5 times faster than EfficientNet.

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeSt50_fast_1s1x64d        | 0.8035 | 0.9528|  0.8035 |            -| 8.68     | 26.3   |
| ResNeSt50        | 0.8083 | 0.9542|  0.8113 |            -| 10.78     | 27.5   |
| RegNetX_4GF        | 0.7850 | 0.9416|  0.7860 |            -| 8.0     | 22.1   |

<a name='3'></a>
## 3. Inference speed based on T4 GPU

| Models             | Crop Size | Resize Short Size | FP16<br>Batch Size=1<br>(ms) | FP16<br>Batch Size=4<br>(ms) | FP16<br>Batch Size=8<br>(ms) | FP32<br>Batch Size=1<br>(ms) | FP32<br>Batch Size=4<br>(ms) | FP32<br>Batch Size=8<br>(ms) |
|--------------------|-----------|-------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| ResNeSt50_fast_1s1x64d          | 224       | 256   | 3.46466           | 5.56647           | 9.11848          | 3.45405      |   8.72680    |    15.48710     |
| ResNeSt50         | 224       | 256               | 7.05851           | 8.97676            | 13.34704          | 6.16248      |   12.0633    |    21.49936     |
| RegNetX_4GF | 224       | 256       | 6.69042    | 8.01664            | 11.60608       | 6.46478     |   11.19862    |    16.89089    |
