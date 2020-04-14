
# 一、模型压缩方法简介

近年来，深度神经网络在计算机视觉、自然语言处理等领域被验证是一种及其有效的解决问题的方法。通过构建合适的神经网络，加以训练，最终网络模型的性能指标基本上都会远远超过传统算法。

在数据量足够大的情况下，通过合理构建网络模型的方式增加其参数量，可以显著改善模型性能，但是这又带来了模型复杂度急剧提升的问题。大模型在实际场景中使用的成本较高。

深度神经网络一般有较多的参数冗余，目前有几种主要的方法对模型进行压缩，减小其参数量。如裁剪、量化、蒸馏等。其中知识蒸馏是指使用教师模型(teacher model)去指导学生模型(student model)学习特定任务，保证小模型在参数量不变的情况下，得到比较大的性能提升，甚至获得与大模型相似的精度指标[1]。PaddleClas融合已有的蒸馏方法[2,3]，提供了一种简单的半监督标签知识蒸馏方案（SSLD，Simple Semi-supervised Label Distillation），基于ImageNet1k分类数据集，在ResNet50_vd以及MobileNet系列上的精度指标如下图所示。


![ssld_performance][ssld_performance]


# 二、SSLD 蒸馏策略

## 2.1 简介

PaddleClas提供了一种简单的半监督标签知识蒸馏方案（SSLD，Simple Semi-supervised Label Distillation），SSLD的流程图如下图所示。

![epic_distillation_framework][distillation_framework]

使用ImageNet22k数据扩充训练集进行训练，使用教师模型的输出软标签作为监督，计算教师模型和学生模型的JS散度，作为训练的损失函数。具体地，有以下主要特色。

* 无需数据集的真值标签，很容易扩展训练集。在SSLD的loss在计算过程中，仅涉及到教师和学生模型对于相同图片的处理结果（经过softmax激活函数处理之后的soft label），因此即使图片数据不包含真值标签，也可以用来进行训练并提升模型性能。该蒸馏方案的无标签蒸馏策略也大大提升了学生模型的性能上限，在数据选择部分会有更加详细的训练策略以及对应的性能指标介绍。


* 教师模型的选择。在进行知识蒸馏时，如果教师模型与学生模型的差异太大，蒸馏得到的结果反而不会有太大收益。教师模型与学生模型的模型性能差异对结果也有很大影响。在蒸馏ResNet50vd模型时，我们选择ResNeXt101_32x16d_wsl模型作为教师模型；在蒸馏MobileNet系列模型时，我们使用蒸馏得到的ResNet50vd模型作为教师模型。


* 改进loss计算方法。分类loss计算最常用的方法就是cross entropy loss，我们经过实验发现，在使用soft label进行训练时，相对于cross entropy loss，KL div loss对模型性能提升几乎无帮助，但是使用具有对称特特性的JS div loss时，在多个蒸馏任务上相比cross entropy loss均有0.2%左右的收益，SSLD中也基于JS div loss展开实验。


* SSLD方案简单，对标签数据几乎无依赖，也便于后续开发。



## 2.2 数据选择


* SSLD蒸馏方案的一大特色就是无需使用图像的真值标签，因此可以任意扩展数据集的大小，考虑到计算资源的限制，我们在这里仅基于ImageNet22k数据集对蒸馏任务的训练集进行扩充。在SSLD蒸馏任务中，我们使用了`Top-k per class`的数据采样方案[3]。具体步骤如下。
    * 训练集去重。我们首先基于Sift特征相似度匹配的方式对ImageNet22k数据集与ImageNet1k验证集进行去重，防止添加的ImageNet22k训练集中包含ImageNet1k验证集图像，最终去除了4511张相似图片。过滤的部分相似图片如下所示。

    ![22k-1k val相似图片][22k_1k_similar_w_sift]

    * 大数据集soft label获取，对于去重后的ImageNet22k数据集，我们使用教师模型进行预测，得到每张图片的soft label。
    * Top-k数据选择，ImageNet1k数据共有1000类，对于每一类，找出属于该类并且得分最高的k张图片，最终得到一个数据量不超过`1000*k`的数据集（某些类上得到的图片数量可能少于k张）。
    * 将该数据集与ImageNet1k的训练集融合组成最终蒸馏模型所使用的数据集，数据量为500W。


# 三、实验

* PaddleClas的蒸馏策略为`大数据集训练+ImageNet1k finetune`的策略。选择合适的教师模型，首先在挑选得到的500W数据集上进行训练，然后在ImageNet1k训练集上进行finetune，最终得到蒸馏后的学生模型。

## 3.1 教师模型的选择

为了验证教师模型和学生模型的模型大小差异和教师模型的模型精度对蒸馏结果的影响，我们做了几组实验验证。训练策略统一为：`cosine_decay_warmup，lr=1.3, epoch=120, bs=2048`，学生模型均为从头训练。

|Teacher Model | Teacher Top1 | Student Model | Student Top1|
|- |:-: |:-: | :-: |
| ResNeXt101_32x16d_wsl | 84.2% | MobileNetV3_large_x1_0 | 75.78% |
| ResNet50_vd | 79.12% | MobileNetV3_large_x1_0 | 75.60% |
| ResNet50_vd | 82.35% | MobileNetV3_large_x1_0 | 76.00% |


从表中可以看出

> 教师模型结构相同时，其精度越高，最终的蒸馏效果也会更好一些。
>
> 教师模型与学生模型的模型大小差异不宜过大，否则反而会影响蒸馏结果的精度。


因此最终在蒸馏实验中，对于ResNet系列学生模型，我们使用`ResNeXt101_32x16d_wsl`作为教师模型；对于MobileNet系列学生模型，我们使用蒸馏得到的`ResNet50_vd`作为教师模型。

## 3.2 大数据蒸馏

基于PaddleClas的蒸馏策略为`大数据集训练+imagenet1k finetune`的策略。

针对从ImageNet22k挑选出的400W数据，融合imagenet1k训练集，组成共500W的训练集进行训练，具体地，在不同模型上的训练超参及效果如下。


|Student Model | num_epoch  | l2_ecay | batch size/gpu cards |  base lr | learning rate decay | top1 acc |
| - |:-: |:-: | :-: |:-: |:-: |:-: |
| MobileNetV1 | 360 | 3e-5 | 4096/8  | 1.6 | cosine_decay_warmup | 77.65% |
| MobileNetV2 | 360 | 1e-5 | 3072/8  | 0.54 | cosine_decay_warmup | 76.34% |
| MobileNetV3_large_x1_0 | 360 | 1e-5 |  5760/24 | 3.65625 | cosine_decay_warmup | 78.54% |
| MobileNetV3_small_x1_0 | 360 | 1e-5 |  5760/24 | 3.65625 | cosine_decay_warmup | 70.11% |
| ResNet50_vd | 360 | 7e-5 | 1024/32 | 0.4 | cosine_decay_warmup | 82.07% |
| ResNet101_vd | 360 | 7e-5 | 1024/32 | 0.4 | cosine_decay_warmup | 83.41% |

## 3.3 ImageNet1k训练集finetune

对于在大数据集上训练的模型，其学习到的特征可能与ImageNet1k数据特征有偏，因此在这里使用ImageNet1k数据集对模型进行finetune。finetune的超参和finetune的精度收益如下。


|Student Model | num_epoch  | l2_ecay | batch size/gpu cards |  base lr | learning rate decay |  top1 acc |
| - |:-: |:-: | :-: |:-: |:-: |:-: |
| MobileNetV1 | 30 | 3e-5 | 4096/8 | 0.016 | cosine_decay_warmup | 77.89%  |
| MobileNetV2 | 360 | 1e-5 | 3072/8  | 0.0054 | cosine_decay_warmup | 76.73% |
| MobileNetV3_large_x1_0 | 30 | 1e-5 |  2048/8 | 0.008 | cosine_decay_warmup | 78.96% |
| MobileNetV3_small_x1_0 | 30 | 1e-5 |  6400/32 | 0.025 | cosine_decay_warmup | 71.28% |
| ResNet50_vd | 60 | 7e-5 | 1024/32 | 0.004 | cosine_decay_warmup | 82.39% |
| ResNet101_vd | 30 | 7e-5 | 1024/32 | 0.004 | cosine_decay_warmup | 83.73% |


## 3.4 实验过程中的一些tricks

### 3.4.1 bn的计算方法

* 在预测过程中，batch norm的平均值与方差是通过加载预训练模型得到（设其模式为test mode），在训练过程中，batch norm是通过统计当前batch的信息（设其模式为test mode），与历史保存信息进行滑动平均计算得到，在蒸馏任务中，我们发现通过train mode，即教师模型的bn实时变化的模式，去指导学生模型，比通过test mode蒸馏，得到的学生模型性能更好一些，下面是一组实验结果，因此我们在该蒸馏方案中，均使用train mode去得到教师模型的soft label。

|Teacher Model | Teacher Top1 | Student Model | Student Top1|
|- |:-: |:-: | :-: |
| ResNet50_vd | 82.35% | MobileNetV3_large_x1_0 | 76.00% |
| ResNet50_vd | 82.35% | MobileNetV3_large_x1_0 | 75.84% |

### 3.4.2 模型名字冲突问题的解决办法
* 在蒸馏过程中，如果遇到命名冲突的问题，如使用ResNet50_vd蒸馏ResNet34_vd，此时直接训练，会提示相同variable名称不匹配的问题，此时可以通过给学生模型或者教师模型中的变量名添加名称的方式解决该问题，如下所示。在训练之后也可以直接根据后缀区分学生模型和教师模型各自包含的参数。

```python
def net(self, input, class_dim=1000):
    student = ResNet34_vd(postfix_name="_student")
    out_student = student.net( input, class_dim=class_dim )

    teacher = ResNet50_vd()
    out_teacher = teacher.net( input, class_dim=class_dim )
    out_teacher.stop_gradient = True

    return  out_teacher, out_student
```

* 训练完成后，可以通过批量重名的方式修改学生模型的参数，以上述代码为例，批量重命名的命令行如下。
```shell
cd model_final # enter model dir
for var in ./*_student; do cp "$var" "../student_model/${var%_student}"; done # batch copy and rename
```

## 四、蒸馏模型的应用

### 4.1 使用方法

* 中间层学习率调整。蒸馏得到的模型的中间层特征图更加精细化，因此将蒸馏模型预训练应用到其他任务中时，如果采取和之前相同的学习率，容易破坏中间层特征。而如果降低整体模型训练的学习率，则会带来训练收敛速度更慢的问题。因此我们使用了中间层学习率调整的策略。具体地：
    * 针对ResNet50_vd，我们设置一个学习率倍数列表，res block之前的3个conv2d卷积参数具有统一的学习率倍数，4个res block的conv2d分别有一个学习率参数，共需设置5个学习率倍数的超参。在实验中发现。用于迁移学习finetune分类模型时，`[0.1,0.1,0.2,0.2,0.3]`的中间层学习率倍数设置在绝大多数的任务中都性能更好；而在目标检测任务中，`[0.05,0.05,0.05,0.1,0.15]`的中间层学习率倍数设置能够带来更大的精度收益。
    * 对于MoblileNetV3_large_1x0，由于其包含15个block，我们设置每3个block共享一个学习率倍数参数，因此需要共5个学习率倍数的参数，最终发现在分类和检测任务中，`[0.25,0.25,0.5,0.5,0.75]`的中间层学习率倍数能够带来更大的精度收益。


* 适当的l2 decay。不同分类模型在训练的时候一般都会根据模型就与设置不同的l2 decay，大模型为了防止过拟合，往往会设置更大的l2 decay，如ResNet50等模型，一般设置为`1e-4`或者`7e-5`(结合数据增广训练可以设置为如此)；而如MobileNet系列模型，在训练时往往都会设置为`1e-5~4e-5`，防止模型过度欠拟合，在蒸馏时亦是如此。在将蒸馏模型应用到目标检测任务中时，我们发现也需要调节backbone甚至特定任务模型模型的l2 decay，和预训练蒸馏时的l2 decay尽可能保持一致。以Faster RCNN MobiletNetV3 FPN为例，我们发现仅修改该参数，在COCO2017数据集上就可以带来最多0.5%左右的精度(mAP)提升（默认Faster RCNN l2 decay为1e-4，我们修改为1e-5~4e-5均有0.3%~0.5%的提升）。


#### 4.2 迁移学习finetune
* 为验证迁移学习的效果，我们在10个小的数据集上验证其效果。在这里为了保证实验的可对比性，我们均使用ImageNet1k数据集训练的标准预处理过程，对于蒸馏模型我们也添加了蒸馏模型中间层学习率的搜索。
* 对于ResNet50_vd，baseline为Top1 Acc 79.12%的预训练模型基于线性grid search搜索得到的最佳精度，对比实验则为基于该精度对预训练和中间层学习率进一步搜索得到的最佳精度。下面给出10个数据集上所有baseline和蒸馏模型的精度对比。


| Dataset | Model | Baseline Top1 Acc | Distillation Model Finetune |
|- |:-: |:-: | :-: |
| Oxford102 flowers | ResNete50_vd | 97.18% | 97.41% |
| caltech-101 | ResNete50_vd | 92.57% | 93.21% |
| Oxford-IIIT-Pets | ResNete50_vd | 94.30% | 94.76% |
| DTD | ResNete50_vd | 76.48% | 77.71% |
| fgvc-aircraft-2013b | ResNete50_vd | 88.98% | 90.00% |
| Stanford-Cars | ResNete50_vd | 92.65% | 92.76% |
| SUN397 | ResNete50_vd | 64.02% | 68.36% |
| cifar100 | ResNete50_vd | 86.50% | 87.58% |
| cifar10 | ResNete50_vd | 97.72% | 97.94% |
| Food-101 | ResNete50_vd | 89.58% | 89.99% |

* 可以看出在上面10个数据集上，结合适当的中间层学习率倍数设置，蒸馏模型平均能够带来1%以上的精度提升。

* 此外，我们也基于MobileNetV3_large_1x0模型验证了蒸馏模型在迁移学习finetune任务中的效果。在这里baseline为Top1 Acc 75.20%的MobileNetV3_large_1x0模型，蒸馏模型为78.16%的蒸馏模型。由于资源限制，我们在这里仅针对上面的前6个数据集展开了实验，实验对比结果如下。

| Dataset | Model | Baseline Top1 Acc | Distillation Model Finetune |
|- |:-: |:-: | :-: |
| Oxford102 flowers | MobileNetV3_large_1x0 | 96.45% | 97.31% |
| caltech-101 | MobileNetV3_large_1x0 | 88.89% | 89.26% |
| Oxford-IIIT-Pets | MobileNetV3_large_1x0 | 91.96% | 92.72% |
| DTD | MobileNetV3_large_1x0 | 73.67% | 74.52% |
| fgvc-aircraft-2013b | MobileNetV3_large_1x0 | 83.37% | 84.75% |
| Stanford-Cars | MobileNetV3_large_1x0 | 89.57% | 90.37% |

* 由上表可以看出蒸馏的MobileNetV3_large_1x0模型相对于baseline，在6个数据集上平均也有0.84%的精度提升。


### 4.3 目标检测

我们基于两阶段目标检测Faster/Cascade RCNN模型验证蒸馏得到的预训练模型的效果。

* ResNet50_vd

设置训练与评测的尺度均为640x640，最终COCO上检测指标如下。

| Model | train/test scale | pretrain top1 acc | feature map lr | coco mAP |
|- |:-: |:-: | :-: | :-: |
| Faster RCNN R50_vd FPN | 640/640 | 79.12% | [1.0,1.0,1.0,1.0,1.0] | 34.9% |
| Faster RCNN R50_vd FPN | 640/640 | 79.80% | [0.05,0.05,0.1,0.1,0.15] | 34.3% |
| Faster RCNN R50_vd FPN | 640/640 | 82.18% | [0.05,0.05,0.1,0.1,0.15] | 36.3% |

在这里可以看出，对于未蒸馏模型，过度调整中间层学习率反而降低最终检测模型的性能指标。基于该蒸馏模型，我们也给出了领先的服务端实用目标检测方案，详细的配置与训练代码均已开源，可以参考[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_server_side_det)。

* MobileNetV3_large_x1_0

同上，我们将backbone修改为MobileNetV3_large_1x0，在这里我们将训练和测试尺度都减小为320x320，调整至合适的训练和策略，最终COCO上检测指标如下。


| Model | train/test scale | pretrain top1 acc | feature map lr | coco mAP |
|- |:-: |:-: | :-: | :-: |
| Cascade RCNN MobileNetV3 FPN | 320/320 | 75.20% | [1.0,1.0,1.0,1.0,1.0] | 22.8% |
| Cascade RCNN MobileNetV3 FPN | 320/320 | 78.96% | [0.25,0.25,0.5,0.5,0.75] | 23.6% |

可以明显看出，蒸馏预训练模型在目标检测中的精度优势也更加明显。结合`CosineDecay`余弦学习率变化策略以及多尺度训练等技巧，最终基于SSLD预训练模型的RCNN检测模型可以在骁龙845芯片上在预测耗时为122ms/image的情况下，COCO mAP达到25.0%，后续也会在[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/master)开源，敬请期待。


### 参考文献

[1] Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network[J]. arXiv preprint arXiv:1503.02531, 2015.

[2] Bagherinezhad H, Horton M, Rastegari M, et al. Label refinery: Improving imagenet classification through label progression[J]. arXiv preprint arXiv:1805.02641, 2018.

[3] Yalniz I Z, Jégou H, Chen K, et al. Billion-scale semi-supervised learning for image classification[J]. arXiv preprint arXiv:1905.00546, 2019.

[4] Xie Q, Hovy E, Luong M T, et al. Self-training with Noisy Student improves ImageNet classification[J]. arXiv preprint arXiv:1911.04252, 2019.




[ssld_performance]: ../../../images/distillation/distillation_perform.png
[22k_1k_similar_w_sift]: ../../../images/distillation/22k_1k_val_compare_w_sift.png
[distillation_framework]: ../../../images/distillation/ppcls_distillation.png
