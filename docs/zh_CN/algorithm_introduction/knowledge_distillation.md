# 知识蒸馏
---
## 目录

* [1. 模型压缩和知识蒸馏方法简介](#1)
* [2. 知识蒸馏应用](#2)
* [3. 知识蒸馏算法介绍](#3)
	* [3.1 Response based distillation](#3.1)
	* [3.2 Feature based distillation](#3.2)
	* [3.3 Relation based distillation](#3.3)
* [4. 参考文献](#4)
<a name='1'></a>
## 1. 模型压缩和知识蒸馏方法简介

近年来，深度神经网络在计算机视觉、自然语言处理等领域被验证是一种极其有效的解决问题的方法。通过构建合适的神经网络，加以训练，最终网络模型的性能指标基本上都会超过传统算法。

在数据量足够大的情况下，通过合理构建网络模型的方式增加其参数量，可以显著改善模型性能，但是这又带来了模型复杂度急剧提升的问题。大模型在实际场景中使用的成本较高。

深度神经网络一般有较多的参数冗余，目前有几种主要的方法对模型进行压缩，减小其参数量。如裁剪、量化、知识蒸馏等，其中知识蒸馏是指使用教师模型(teacher model)去指导学生模型(student model)学习特定任务，保证小模型在参数量不变的情况下，得到比较大的性能提升，甚至获得与大模型相似的精度指标 [1]。

目前知识蒸馏的方法大致可以分为以下三种。

* Response based distillation：教师模型对学生模型的输出进行监督。
* Feature based distillation：教师模型对学生模型的中间层 feature map 进行监督。
* Relation based distillation：对于不同的样本，使用教师模型和学生模型同时计算样本之间 feature map 的相关性，使得学生模型和教师模型得到的相关性矩阵尽可能一致。

<a name='2'></a>
## 2. 知识蒸馏应用


知识蒸馏算法在模型轻量化过程任务中应用广泛，对于需要满足特定的精度的任务，通过使用知识蒸馏的方法，我们可以使用更小的模型便能达到要求的精度，从而减小了模型部署的成本。

此外，对于相同的模型结构，使用知识蒸馏训练得到的预训练模型精度往往更高，这些预训练模型往往也可以提升下游任务的模型精度。比如在图像分类任务中，基于知识蒸馏算法得到的精度更高的预训练模型，也能够在目标检测、图像分割、OCR、视频分类等任务中获得明显的精度收益。


<a name='3'></a>
## 3. 知识蒸馏算法介绍
<a name='3.1'></a>
### 3.1 Response based distillation

最早的知识蒸馏算法 KD，由 Hinton 提出，训练的损失函数中除了 gt loss 之外，还引入了学生模型与教师模型输出的 KL 散度，最终精度超过单纯使用 gt loss 训练的精度。这里需要注意的是，在训练的时候，需要首先训练得到一个更大的教师模型，来指导学生模型的训练过程。

PaddleClas 中提出了一种简单使用的 SSLD 知识蒸馏算法 [6]，在训练的时候去除了对 gt label 的依赖，结合大量无标注数据，最终蒸馏训练得到的预训练模型在 15 个模型上的精度提升平均高达 3%。

上述标准的蒸馏方法是通过一个大模型作为教师模型来指导学生模型提升效果，而后来又发展出 DML(Deep Mutual Learning)互学习蒸馏方法 [7]，即通过两个结构相同的模型互相学习。具体的。相比于 KD 等依赖于大的教师模型的知识蒸馏算法，DML 脱离了对大的教师模型的依赖，蒸馏训练的流程更加简单，模型产出效率也要更高一些。

<a name='3.2'></a>
### 3.2 Feature based distillation

Heo 等人提出了 OverHaul [8], 计算学生模型与教师模型的 feature map distance，作为蒸馏的 loss，在这里使用了学生模型、教师模型的转移，来保证二者的 feature map 可以正常地进行 distance 的计算。

基于 feature map distance 的知识蒸馏方法也能够和 `3.1 章节` 中的基于 response 的知识蒸馏算法融合在一起，同时对学生模型的输出结果和中间层 feature map 进行监督。而对于 DML 方法来说，这种融合过程更为简单，因为不需要对学生和教师模型的 feature map 进行转换，便可以完成对齐(alignment)过程。PP-OCRv2 系统中便使用了这种方法，最终大幅提升了 OCR 文字识别模型的精度。

<a name='3.3'></a>
### 3.3 Relation based distillation


`3.1` 和 `3.2` 章节中的论文中主要是考虑到学生模型与教师模型的输出或者中间层 feature map，这些知识蒸馏算法只关注个体的输出结果，没有考虑到个体之间的输出关系。

Park 等人提出了 RKD [10]，基于关系的知识蒸馏算法，RKD 中进一步考虑个体输出之间的关系，使用 2 种损失函数，二阶的距离损失（distance-wise）和三阶的角度损失（angle-wise）


本论文提出的算法关系知识蒸馏（RKD）迁移教师模型得到的输出结果间的结构化关系给学生模型，不同于之前的只关注个体输出结果，RKD 算法使用两种损失函数：二阶的距离损失(distance-wise)和三阶的角度损失(angle-wise)。在最终计算蒸馏损失函数的时候，同时考虑 KD loss 和 RKD loss。最终精度优于单独使用 KD loss 蒸馏得到的模型精度。

<a name='4'></a>
## 4. 参考文献

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
