# DPN and DenseNet series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)
* [3. Inference speed based on V100 GPU](#3)
* [4. Inference speed based on T4 GPU](#4)

<a name='1'></a>
## 1. Overview

DenseNet is a new network structure proposed in 2017 and was the best paper of CVPR. The network has designed a new cross-layer connected block called dense-block. Compared to the bottleneck in ResNet, dense-block has designed a more aggressive dense connection module, that is, connecting all the layers to each other, and each layer will accept all the layers in front of it as its additional input. DenseNet stacks all dense-blocks into a densely connected network. The dense connection makes DenseNet easier to backpropagate, making the network easier to train and converge.  The full name of DPN is Dual Path Networks, which is a network composed of DenseNet and ResNeXt, which proves that DenseNet can extract new features from the previous level, and ResNeXt essentially reuses the extracted features . The author further analyzes and finds that ResNeXt has high reuse rate for features, but low redundancy, while DenseNet can create new features, but with high redundancy. Combining the advantages of the two structures, the author designed the DPN network. In the end, the DPN network achieved better results than ResNeXt and DenseNet under the same FLOPs and parameters.

The FLOPs, parameters, and inference time on the T4 GPU of this series of models are shown in the figure below.

![](../../images/models/T4_benchmark/t4.fp32.bs4.DPN.flops.png)

![](../../images/models/T4_benchmark/t4.fp32.bs4.DPN.params.png)

![](../../images/models/T4_benchmark/t4.fp32.bs4.DPN.png)

![](../../images/models/T4_benchmark/t4.fp16.bs4.DPN.png)

The pretrained models of these two types of models (a total of 10) are open sourced in PaddleClas at present. The indicators are shown in the figure above. It is easy to observe that under the same FLOPs and parameters, DPN has higher accuracy than DenseNet. However,because DPN has more branches, its inference speed is slower than DenseNet. Since DenseNet264 has the deepest layers in all DenseNet networks, it has the largest parameters,DenseNet161 has the largest width, resulting the largest FLOPs and the highest accuracy in this series. From the perspective of inference speed, DenseNet161, which has a large FLOPs and high accuracy, has a faster speed than DenseNet264, so it has a greater advantage than DenseNet264.

For DPN series networks, the larger the model's FLOPs and parameters, the higher the model's accuracy. Among them, since the width of DPN107 is the largest, it has the largest number of parameters and FLOPs in this series of networks.

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models      | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DenseNet121 | 0.757  | 0.926  | 0.750             |                   | 5.690        | 7.980             |
| DenseNet161 | 0.786  | 0.941  | 0.778             |                   | 15.490       | 28.680            |
| DenseNet169 | 0.768  | 0.933  | 0.764             |                   | 6.740        | 14.150            |
| DenseNet201 | 0.776  | 0.937  | 0.775             |                   | 8.610        | 20.010            |
| DenseNet264 | 0.780  | 0.939  | 0.779             |                   | 11.540       | 33.370            |
| DPN68       | 0.768  | 0.934  | 0.764             | 0.931             | 4.030        | 10.780            |
| DPN92       | 0.799  | 0.948  | 0.793             | 0.946             | 12.540       | 36.290            |
| DPN98       | 0.806  | 0.951  | 0.799             | 0.949             | 22.220       | 58.460            |
| DPN107      | 0.809  | 0.953  | 0.802             | 0.951             | 35.060       | 82.970            |
| DPN131      | 0.807  | 0.951  | 0.801             | 0.949             | 30.510       | 75.360            |



<a name='3'></a>
## 3. Inference speed based on V100 GPU

| Models                               | Crop Size | Resize Short Size | FP32<br>Batch Size=1<br>(ms) |
|-------------|-----------|-------------------|--------------------------|
| DenseNet121 | 224       | 256               | 4.371                    |
| DenseNet161 | 224       | 256               | 8.863                    |
| DenseNet169 | 224       | 256               | 6.391                    |
| DenseNet201 | 224       | 256               | 8.173                    |
| DenseNet264 | 224       | 256               | 11.942                   |
| DPN68       | 224       | 256               | 11.805                   |
| DPN92       | 224       | 256               | 17.840                   |
| DPN98       | 224       | 256               | 21.057                   |
| DPN107      | 224       | 256               | 28.685                   |
| DPN131      | 224       | 256               | 28.083                   |


<a name='4'></a>
## 4. Inference speed based on T4 GPU

| Models      | Crop Size | Resize Short Size | FP16<br>Batch Size=1<br>(ms) | FP16<br>Batch Size=4<br>(ms) | FP16<br>Batch Size=8<br>(ms) | FP32<br>Batch Size=1<br>(ms) | FP32<br>Batch Size=4<br>(ms) | FP32<br>Batch Size=8<br>(ms) |
|-------------|-----------|-------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| DenseNet121 | 224       | 256               | 4.16436                      | 7.2126                       | 10.50221                     | 4.40447                      | 9.32623                      | 15.25175                     |
| DenseNet161 | 224       | 256               | 9.27249                      | 14.25326                     | 20.19849                     | 10.39152                     | 22.15555                     | 35.78443                     |
| DenseNet169 | 224       | 256               | 6.11395                      | 10.28747                     | 13.68717                     | 6.43598                      | 12.98832                     | 20.41964                     |
| DenseNet201 | 224       | 256               | 7.9617                       | 13.4171                      | 17.41949                     | 8.20652                      | 17.45838                     | 27.06309                     |
| DenseNet264 | 224       | 256               | 11.70074                     | 19.69375                     | 24.79545                     | 12.14722                     | 26.27707                     | 40.01905                     |
| DPN68       | 224       | 256               | 11.7827                      | 13.12652                     | 16.19213                     | 11.64915                     | 12.82807                     | 18.57113                     |
| DPN92       | 224       | 256               | 18.56026                     | 20.35983                     | 29.89544                     | 18.15746                     | 23.87545                     | 38.68821                     |
| DPN98       | 224       | 256               | 21.70508                     | 24.7755                      | 40.93595                     | 21.18196                     | 33.23925                     | 62.77751                     |
| DPN107      | 224       | 256               | 27.84462                     | 34.83217                     | 60.67903                     | 27.62046                     | 52.65353                     | 100.11721                    |
| DPN131      | 224       | 256               | 28.58941                     | 33.01078                     | 55.65146                     | 28.33119                     | 46.19439                     | 89.24904                     |
