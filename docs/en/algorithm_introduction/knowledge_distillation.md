# Knowledge Distillation

---
## Content

* [1. Introduction of model compression methods](#1)
* [2. Application of knowledge distillation](#2)
* [3. Overview of knowledge distillation methods](#3)
    * [3.1 Response based distillation](#3.1)
    * [3.2 Feature based distillation](#3.2)
    * [3.3 Relation based distillation](#3.3)
* [4. Reference](#4)

<a name='1'></a>

## 1. Introduction of model compression methods

In recent years, deep neural networks have been proved to be an extremely effective method for solving problems in the fields of computer vision and natural language processing. A suitable neural network architecture might performs better than traditional algorithms mostly.

When the amount of data is large enough, increasing the model parameters with a reasonable method can significantly improve the model performance, but this brings about the problem of a sharp increase of the model complexity. It costs more for larger models.

Parameter redundancy exists in deep neural networks generally. At present, there are several mainstream methods to compress the model and reduce parameters. Such as pruning, quantization, knowledge distillation, etc. Knowledge distillation refers to the use of a teacher model to guide the student model to learn specific tasks to ensure that the small model obtains relatively large performance, and even has comparable performance with the large model [1].


Currently, knowledge distillation methods can be roughly divided into the following three types.

* Response based distillation: Output of student model is guided by the teacher model for
* Feature based distillation: Inner feature map of student model is guided by the teacher model.
* Relation based distillation: For different samples, the teacher model and the student model are used to calculate the correlation of the feature map between the samples, the final goal is to make sure that correlation matrix of student model and the teacher model are as consistent as possible.


<a name='2'></a>

## 2. Application of knowledge distillation

Knowledge distillation algorithm is widely used in lightweight tasks. For tasks that need to meet specific accuracy, by using the knowledge distillation method, we can achieve the required accuracy with a smaller model, thereby reducing model deployment cost.


What's more, for the same model structure, pre-trained models obtained by knowledge distillation often performs better, and these pre-trained models can also improve performance of the downstream tasks. For example, a pre-trained image classification model with higher accuracy can also help other tasks obtain significant accuracy gains such as target detection, image segmentation, OCR, and video classification.

<a name='3'></a>

## 3. Overview of knowledge distillation methods

<a name='3.1'></a>

### 3.1 Response based distillation


Knowledge distillation algorithm is firstly proposed by Hinton, which is called KD. In addition to base cross entropy loss, KL divergence loss between output of student model and teacher model is also added into the total training loss. It's noted that a larger teacher model is needed to guide the training process of the student model.

PaddleClas proposed a simple but useful knowledge distillation algorithm canlled SSLD [6], Labels are not needed for SSLD, so unlabeled data can also be used for training. Accuracy of 15 models has more than 3% improvement using SSLD.

Teacher model is needed for the above-mentioned distillation method to guide the student model training process. Deep Mutual Learning (DML) is then proposed [7], for which two models with same architecture learn from each other to obtain higher accuracy. Compared with KD and other knowledge distillation algorithms that rely on large teacher models, DML is free of dependence on large teacher models. The distillation training process is simpler.

<a name='3.2'></a>

### 3.2 Feature based distillation

Heo et al. proposed OverHaul [8], which calculates the feature map distance between the student model and the teacher model, as distillation loss. Here, feature map alignment of the student model and the teacher model is used to ensure that the feature maps' distance can be calculated.

Feature based distillation can also be integrated with the response based knowledge distillation algorithm in Chapter 3.1, which means both the inner feature map and output of the student model are guided during the training process. For the DML method, this integration process is simpler, because the alignment process is not needed since the two models' architectures are absolutely same. This integration process is used in the PP-OCRv2 system, which ultimately greatly improves the accuracy of the OCR text recognition model.

<a name='3.3'></a>

### 3.3 Relation based distillation

The papers in chapters `3.1` and `3.2` mainly consider the inner feature map or final output of the student model and the teacher model. These knowledge distillation algorithms only focus on the output for single sample, but do not consider the output relationship between different samples.

Park et al. proposed RKD [10], a relationship-based knowledge distillation algorithm. In RKD, the relationship between different samples is further considered, and two loss functions are used, which are the second-order distance loss (distance-wise) and the third-order angle loss (angle-wise). For the final distillation loss, KD loss and RKD loss are considered at the same time. The final accuracy is better than the accuracy of the model obtained just using KD loss.

<a name='4'></a>

## 4. Reference

[1] Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network[J]. arXiv preprint arXiv:1503.02531, 2015.

[2] Bagherinezhad H, Horton M, Rastegari M, et al. Label refinery: Improving imagenet classification through label progression[J]. arXiv preprint arXiv:1805.02641, 2018.

[3] Yalniz I Z, Jégou H, Chen K, et al. Billion-scale semi-supervised learning for image classification[J]. arXiv preprint arXiv:1905.00546, 2019.

[4] Cubuk E D, Zoph B, Mane D, et al. Autoaugment: Learning augmentation strategies from data[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2019: 113-123.

[5] Touvron H, Vedaldi A, Douze M, et al. Fixing the train-test resolution discrepancy[C]//Advances in Neural Information Processing Systems. 2019: 8250-8260.

[6] Cui C, Guo R, Du Y, et al. Beyond Self-Supervision: A Simple Yet Effective Network Distillation Alternative to Improve Backbones[J]. arXiv preprint arXiv:2103.05959, 2021.

[7] Zhang Y, Xiang T, Hospedales T M, et al. Deep mutual learning[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4320-4328.

[8] Heo B, Kim J, Yun S, et al. A comprehensive overhaul of feature distillation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1921-1930.

[9] Du Y, Li C, Guo R, et al. PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System[J]. arXiv preprint arXiv:2109.03144, 2021.

[10] Park W, Kim D, Lu Y, et al. Relational knowledge distillation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 3967-3976.
