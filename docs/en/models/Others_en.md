# Other networks
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)
* [3. Inference speed and storage size based on SD855](#3)
* [4. Inference speed based on T4 GPU](#4)

<a name='1'></a>
## 1. Overview

In 2012, AlexNet network proposed by Alex et al. won the ImageNet competition by far surpassing the second place, and the convolutional neural network and even deep learning attracted wide attention. AlexNet used relu as the activation function of CNN to solve the gradient dispersion problem of sigmoid when the network is deep. During the training, Dropout was used to randomly lose a part of the neurons, avoiding the overfitting of the model. In the network, overlapping maximum pooling is used to replace the average pooling commonly used in CNN, which avoids the fuzzy effect of average pooling and improves the feature richness. In a sense, AlexNet has exploded the research and application of neural networks.

SqueezeNet achieved the same precision as AlexNet on Imagenet-1k, but only with 1/50 parameters. The core of the network is the Fire module, which used the convolution of 1x1 to achieve channel dimensionality reduction, thus greatly saving the number of parameters. The author created SqueezeNet by stacking a large number of Fire modules.

VGG is a convolutional neural network developed by researchers at Oxford University's Visual Geometry Group and DeepMind. The network explores the relationship between the depth of the convolutional neural network and its performance. By repeatedly stacking the small convolutional kernel of 3x3 and the maximum pooling layer of 2x2, the multi-layer convolutional neural network is successfully constructed and has achieved good convergence accuracy. In the end, VGG won the runner-up of ILSVRC 2014 classification and the champion of positioning.

DarkNet53 is designed for object detection by YOLO author in the paper. The network is basically composed of 1x1 and 3x3 kernel, with a total of 53 layers, named DarkNet53.


<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

| Models                    | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| AlexNet                   | 0.567  | 0.792  | 0.5720            |                   | 1.370        | 61.090            |
| SqueezeNet1_0             | 0.596  | 0.817  | 0.575             |                   | 1.550        | 1.240             |
| SqueezeNet1_1             | 0.601  | 0.819  |                   |                   | 0.690        | 1.230             |
| VGG11                     | 0.693  | 0.891  |                   |                   | 15.090       | 132.850           |
| VGG13                     | 0.700  | 0.894  |                   |                   | 22.480       | 133.030           |
| VGG16                     | 0.720  | 0.907  | 0.715             | 0.901             | 30.810       | 138.340           |
| VGG19                     | 0.726  | 0.909  |                   |                   | 39.130       | 143.650           |
| DarkNet53                 | 0.780  | 0.941  | 0.772             | 0.938             | 18.580       | 41.600            |


<a name='3'></a>
## 3. Inference speed based on V100 GPU


| Models                 | Crop Size | Resize Short Size | FP32<br>Batch Size=1<br>(ms) |
|---------------------------|-----------|-------------------|----------------------|
| AlexNet                   | 224       | 256               | 1.176                |
| SqueezeNet1_0             | 224       | 256               | 0.860                |
| SqueezeNet1_1             | 224       | 256               | 0.763                |
| VGG11                     | 224       | 256               | 1.867                |
| VGG13                     | 224       | 256               | 2.148                |
| VGG16                     | 224       | 256               | 2.616                |
| VGG19                     | 224       | 256               | 3.076                |
| DarkNet53                 | 256       | 256               | 3.139                |

<a name='4'></a>
## 4. Inference speed based on T4 GPU

| Models                | Crop Size | Resize Short Size | FP16<br>Batch Size=1<br>(ms) | FP16<br>Batch Size=4<br>(ms) | FP16<br>Batch Size=8<br>(ms) | FP32<br>Batch Size=1<br>(ms) | FP32<br>Batch Size=4<br>(ms) | FP32<br>Batch Size=8<br>(ms) |
|-----------------------|-----------|-------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| AlexNet               | 224       | 256               | 1.06447                      | 1.70435                      | 2.38402                      | 1.44993                      | 2.46696                      | 3.72085                      |
| SqueezeNet1_0         | 224       | 256               | 0.97162                      | 2.06719                      | 3.67499                      | 0.96736                      | 2.53221                      | 4.54047                      |
| SqueezeNet1_1         | 224       | 256               | 0.81378                      | 1.62919                      | 2.68044                      | 0.76032                      | 1.877                        | 3.15298                      |
| VGG11                 | 224       | 256               | 2.24408                      | 4.67794                      | 7.6568                       | 3.90412                      | 9.51147                      | 17.14168                     |
| VGG13                 | 224       | 256               | 2.58589                      | 5.82708                      | 10.03591                     | 4.64684                      | 12.61558                     | 23.70015                     |
| VGG16                 | 224       | 256               | 3.13237                      | 7.19257                      | 12.50913                     | 5.61769                      | 16.40064                     | 32.03939                     |
| VGG19                 | 224       | 256               | 3.69987                      | 8.59168                      | 15.07866                     | 6.65221                      | 20.4334                      | 41.55902                     |
| DarkNet53             | 256       | 256               | 3.18101                      | 5.88419                      | 10.14964                     | 4.10829                      | 12.1714                      | 22.15266                     |
