# SEResNeXt and Res2Net series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)
* [3. Inference speed based on V100 GPU](#3)
* [4. Inference speed based on T4 GPU](#4)

<a name='1'></a>
## 1. Overview

ResNeXt, one of the typical variants of ResNet, was presented at the CVPR conference in 2017. Prior to this, the methods to improve the model accuracy mainly focused on deepening or widening the network, which increased the number of parameters and calculation, and slowed down the inference speed accordingly. The concept of cardinality was proposed in ResNeXt structure. The author found that increasing the number of channel groups was more effective than increasing the depth and width through experiments. It can improve the accuracy without increasing the parameter complexity and reduce the number of parameters at the same time, so it is a more successful variant of ResNet.

SENet is the winner of the 2017 ImageNet classification competition. It proposes a new SE structure that can be migrated to any other network. It controls the scale to enhance the important features between each channel, and weaken the unimportant features. So that the extracted features are more directional.

Res2Net is a brand-new improvement of ResNet proposed in 2019. The solution can be easily integrated with other excellent modules. Without increasing the amount of calculation, the performance on ImageNet, CIFAR-100 and other data sets exceeds ResNet. Res2Net, with its simple structure and superior performance, further explores the multi-scale representation capability of CNN at a more fine-grained level. Res2Net reveals a new dimension to improve model accuracy, called scale, which is an essential and more effective factor in addition to the existing dimensions of depth, width, and cardinality. The network also performs well in other visual tasks such as object detection and image segmentation.

The FLOPs, parameters, and inference time on the T4 GPU of this series of models are shown in the figure below.


![](../../images/models/T4_benchmark/t4.fp32.bs4.SeResNeXt.flops.png)

![](../../images/models/T4_benchmark/t4.fp32.bs4.SeResNeXt.params.png)

![](../../images/models/T4_benchmark/t4.fp32.bs4.SeResNeXt.png)

![](../../images/models/T4_benchmark/t4.fp16.bs4.SeResNeXt.png)


At present, there are a total of 24 pretrained models of the three categories open sourced by PaddleClas, and the indicators are shown in the figure. It can be seen from the diagram that under the same Flops and Params, the improved model tends to have higher accuracy, but the inference speed is often inferior to the ResNet series. On the other hand, Res2Net performed better. Compared with group operation in ResNeXt and SE structure operation in SEResNet, Res2Net tended to have better accuracy in the same Flops, Params and inference speed.


<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models                | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Res2Net50_26w_4s      | 0.793  | 0.946  | 0.780             | 0.936             | 8.520        | 25.700            |
| Res2Net50_vd_26w_4s   | 0.798  | 0.949  |                   |                   | 8.370        | 25.060            |
| Res2Net50_vd_26w_4s_ssld   | 0.831  | 0.966  |                   |                   | 8.370        | 25.060            |
| Res2Net50_14w_8s      | 0.795  | 0.947  | 0.781             | 0.939             | 9.010        | 25.720            |
| Res2Net101_vd_26w_4s  | 0.806  | 0.952  |                   |                   | 16.670       | 45.220            |
| Res2Net101_vd_26w_4s_ssld  | 0.839  | 0.971  |                   |                   | 16.670       | 45.220            |
| Res2Net200_vd_26w_4s  | 0.812  | 0.957  |                   |                   | 31.490       | 76.210            |
| Res2Net200_vd_26w_4s_ssld  | **0.851**  | 0.974  |                   |                   | 31.490       | 76.210            |
| ResNeXt50_32x4d       | 0.778  | 0.938  | 0.778             |                   | 8.020        | 23.640            |
| ResNeXt50_vd_32x4d    | 0.796  | 0.946  |                   |                   | 8.500        | 23.660            |
| ResNeXt50_64x4d       | 0.784  | 0.941  |                   |                   | 15.060       | 42.360            |
| ResNeXt50_vd_64x4d    | 0.801  | 0.949  |                   |                   | 15.540       | 42.380            |
| ResNeXt101_32x4d      | 0.787  | 0.942  | 0.788             |                   | 15.010       | 41.540            |
| ResNeXt101_vd_32x4d   | 0.803  | 0.951  |                   |                   | 15.490       | 41.560            |
| ResNeXt101_64x4d      | 0.784  | 0.945  | 0.796             |                   | 29.050       | 78.120            |
| ResNeXt101_vd_64x4d   | 0.808  | 0.952  |                   |                   | 29.530       | 78.140            |
| ResNeXt152_32x4d      | 0.790  | 0.943  |                   |                   | 22.010       | 56.280            |
| ResNeXt152_vd_32x4d   | 0.807  | 0.952  |                   |                   | 22.490       | 56.300            |
| ResNeXt152_64x4d      | 0.795  | 0.947  |                   |                   | 43.030       | 107.570           |
| ResNeXt152_vd_64x4d   | 0.811  | 0.953  |                   |                   | 43.520       | 107.590           |
| SE_ResNet18_vd        | 0.733  | 0.914  |                   |                   | 4.140        | 11.800            |
| SE_ResNet34_vd        | 0.765  | 0.932  |                   |                   | 7.840        | 21.980            |
| SE_ResNet50_vd        | 0.795  | 0.948  |                   |                   | 8.670        | 28.090            |
| SE_ResNeXt50_32x4d    | 0.784  | 0.940  | 0.789             | 0.945             | 8.020        | 26.160            |
| SE_ResNeXt50_vd_32x4d | 0.802  | 0.949  |                   |                   | 10.760       | 26.280            |
| SE_ResNeXt101_32x4d   | 0.7939  | 0.9443  | 0.793             | 0.950             | 15.020       | 46.280            |
| SENet154_vd           | 0.814  | 0.955  |                   |                   | 45.830       | 114.290           |


<a name='3'></a>
## 3. Inference speed based on V100 GPU

| Models                 | Crop Size | Resize Short Size | FP32<br>Batch Size=1<br>(ms) |
|-----------------------|-----------|-------------------|--------------------------|
| Res2Net50_26w_4s      | 224       | 256               | 4.148                    |
| Res2Net50_vd_26w_4s   | 224       | 256               | 4.172                    |
| Res2Net50_14w_8s      | 224       | 256               | 5.113                    |
| Res2Net101_vd_26w_4s  | 224       | 256               | 7.327                    |
| Res2Net200_vd_26w_4s  | 224       | 256               | 12.806                   |
| ResNeXt50_32x4d       | 224       | 256               | 10.964                   |
| ResNeXt50_vd_32x4d    | 224       | 256               | 7.566                    |
| ResNeXt50_64x4d       | 224       | 256               | 13.905                   |
| ResNeXt50_vd_64x4d    | 224       | 256               | 14.321                   |
| ResNeXt101_32x4d      | 224       | 256               | 14.915                   |
| ResNeXt101_vd_32x4d   | 224       | 256               | 14.885                   |
| ResNeXt101_64x4d      | 224       | 256               | 28.716                   |
| ResNeXt101_vd_64x4d   | 224       | 256               | 28.398                   |
| ResNeXt152_32x4d      | 224       | 256               | 22.996                   |
| ResNeXt152_vd_32x4d   | 224       | 256               | 22.729                   |
| ResNeXt152_64x4d      | 224       | 256               | 46.705                   |
| ResNeXt152_vd_64x4d   | 224       | 256               | 46.395                   |
| SE_ResNet18_vd        | 224       | 256               | 1.694                    |
| SE_ResNet34_vd        | 224       | 256               | 2.786                    |
| SE_ResNet50_vd        | 224       | 256               | 3.749                    |
| SE_ResNeXt50_32x4d    | 224       | 256               | 8.924                    |
| SE_ResNeXt50_vd_32x4d | 224       | 256               | 9.011                    |
| SE_ResNeXt101_32x4d   | 224       | 256               | 19.204                   |
| SENet154_vd           | 224       | 256               | 50.406                   |

<a name='4'></a>
## 4. Inference speed based on T4 GPU

| Models                | Crop Size | Resize Short Size | FP16<br>Batch Size=1<br>(ms) | FP16<br>Batch Size=4<br>(ms) | FP16<br>Batch Size=8<br>(ms) | FP32<br>Batch Size=1<br>(ms) | FP32<br>Batch Size=4<br>(ms) | FP32<br>Batch Size=8<br>(ms) |
|-----------------------|-----------|-------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| Res2Net50_26w_4s      | 224       | 256               | 3.56067                      | 6.61827                      | 11.41566                     | 4.47188                      | 9.65722                      | 17.54535                     |
| Res2Net50_vd_26w_4s   | 224       | 256               | 3.69221                      | 6.94419                      | 11.92441                     | 4.52712                      | 9.93247                      | 18.16928                     |
| Res2Net50_14w_8s      | 224       | 256               | 4.45745                      | 7.69847                      | 12.30935                     | 5.4026                       | 10.60273                     | 18.01234                     |
| Res2Net101_vd_26w_4s  | 224       | 256               | 6.53122                      | 10.81895                     | 18.94395                     | 8.08729                      | 17.31208                     | 31.95762                     |
| Res2Net200_vd_26w_4s  | 224       | 256               | 11.66671                     | 18.93953                     | 33.19188                     | 14.67806                     | 32.35032                     | 63.65899                     |
| ResNeXt50_32x4d       | 224       | 256               | 7.61087                      | 8.88918                      | 12.99674                     | 7.56327                      | 10.6134                      | 18.46915                     |
| ResNeXt50_vd_32x4d    | 224       | 256               | 7.69065                      | 8.94014                      | 13.4088                      | 7.62044                      | 11.03385                     | 19.15339                     |
| ResNeXt50_64x4d       | 224       | 256               | 13.78688                     | 15.84655                     | 21.79537                     | 13.80962                     | 18.4712                      | 33.49843                     |
| ResNeXt50_vd_64x4d    | 224       | 256               | 13.79538                     | 15.22201                     | 22.27045                     | 13.94449                     | 18.88759                     | 34.28889                     |
| ResNeXt101_32x4d      | 224       | 256               | 16.59777                     | 17.93153                     | 21.36541                     | 16.21503                     | 19.96568                     | 33.76831                     |
| ResNeXt101_vd_32x4d   | 224       | 256               | 16.36909                     | 17.45681                     | 22.10216                     | 16.28103                     | 20.25611                     | 34.37152                     |
| ResNeXt101_64x4d      | 224       | 256               | 30.12355                     | 32.46823                     | 38.41901                     | 30.4788                      | 36.29801                     | 68.85559                     |
| ResNeXt101_vd_64x4d   | 224       | 256               | 30.34022                     | 32.27869                     | 38.72523                     | 30.40456                     | 36.77324                     | 69.66021                     |
| ResNeXt152_32x4d      | 224       | 256               | 25.26417                     | 26.57001                     | 30.67834                     | 24.86299                     | 29.36764                     | 52.09426                     |
| ResNeXt152_vd_32x4d   | 224       | 256               | 25.11196                     | 26.70515                     | 31.72636                     | 25.03258                     | 30.08987                     | 52.64429                     |
| ResNeXt152_64x4d      | 224       | 256               | 46.58293                     | 48.34563                     | 56.97961                     | 46.7564                      | 56.34108                     | 106.11736                    |
| ResNeXt152_vd_64x4d   | 224       | 256               | 47.68447                     | 48.91406                     | 57.29329                     | 47.18638                     | 57.16257                     | 107.26288                    |
| SE_ResNet18_vd        | 224       | 256               | 1.61823                      | 3.1391                       | 4.60282                      | 1.7691                       | 4.19877                      | 7.5331                       |
| SE_ResNet34_vd        | 224       | 256               | 2.67518                      | 5.04694                      | 7.18946                      | 2.88559                      | 7.03291                      | 12.73502                     |
| SE_ResNet50_vd        | 224       | 256               | 3.65394                      | 7.568                        | 12.52793                     | 4.28393                      | 10.38846                     | 18.33154                     |
| SE_ResNeXt50_32x4d    | 224       | 256               | 9.06957                      | 11.37898                     | 18.86282                     | 8.74121                      | 13.563                       | 23.01954                     |
| SE_ResNeXt50_vd_32x4d | 224       | 256               | 9.25016                      | 11.85045                     | 25.57004                     | 9.17134                      | 14.76192                     | 19.914                       |
| SE_ResNeXt101_32x4d   | 224       | 256               | 19.34455                     | 20.6104                      | 32.20432                     | 18.82604                     | 25.31814                     | 41.97758                     |
| SENet154_vd           | 224       | 256               | 49.85733                     | 54.37267                     | 74.70447                     | 53.79794                     | 66.31684                     | 121.59885                    |
