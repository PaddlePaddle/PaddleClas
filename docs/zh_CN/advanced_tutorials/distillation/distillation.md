
# 一、模型压缩方法简介

近年来，深度神经网络在计算机视觉、自然语言处理等领域被验证是一种极其有效的解决问题的方法。通过构建合适的神经网络，加以训练，最终网络模型的性能指标基本上都会超过传统算法。

在数据量足够大的情况下，通过合理构建网络模型的方式增加其参数量，可以显著改善模型性能，但是这又带来了模型复杂度急剧提升的问题。大模型在实际场景中使用的成本较高。

深度神经网络一般有较多的参数冗余，目前有几种主要的方法对模型进行压缩，减小其参数量。如裁剪、量化、知识蒸馏等，其中知识蒸馏是指使用教师模型(teacher model)去指导学生模型(student model)学习特定任务，保证小模型在参数量不变的情况下，得到比较大的性能提升，甚至获得与大模型相似的精度指标[1]。PaddleClas融合已有的蒸馏方法[2,3]，提供了一种简单的半监督标签知识蒸馏方案（SSLD，Simple Semi-supervised Label Distillation），基于ImageNet1k分类数据集，在ResNet_vd以及MobileNet系列上的精度均有超过3%的绝对精度提升，具体指标如下图所示。


![](../../../images/distillation/distillation_perform_s.jpg)


# 二、SSLD 蒸馏策略

## 2.1 简介

SSLD的流程图如下图所示。

![](../../../images/distillation/ppcls_distillation.png)

首先，我们从ImageNet22k中挖掘出了近400万张图片，同时与ImageNet-1k训练集整合在一起，得到了一个新的包含500万张图片的数据集。然后，我们将学生模型与教师模型组合成一个新的网络，该网络分别输出学生模型和教师模型的预测分布，与此同时，固定教师模型整个网络的梯度，而学生模型可以做正常的反向传播。最后，我们将两个模型的logits经过softmax激活函数转换为soft label，并将二者的soft label做JS散度作为损失函数，用于蒸馏模型训练。下面以MobileNetV3（该模型直接训练，精度为75.3%）的知识蒸馏为例，介绍该方案的核心关键点（baseline为79.12%的ResNet50_vd模型蒸馏MobileNetV3，训练集为ImageNet1k训练集，loss为cross entropy loss，迭代轮数为120epoch，精度指标为75.6%）。

* 教师模型的选择。在进行知识蒸馏时，如果教师模型与学生模型的结构差异太大，蒸馏得到的结果反而不会有太大收益。相同结构下，精度更高的教师模型对结果也有很大影响。相比于79.12%的ResNet50_vd教师模型，使用82.4%的ResNet50_vd教师模型可以带来0.4%的绝对精度收益(`75.6%->76.0%`)。

* 改进loss计算方法。分类loss计算最常用的方法就是cross entropy loss，我们经过实验发现，在使用soft label进行训练时，相对于cross entropy loss，KL div loss对模型性能提升几乎无帮助，但是使用具有对称特性的JS div loss时，在多个蒸馏任务上相比cross entropy loss均有0.2%左右的收益(`76.0%->76.2%`)，SSLD中也基于JS div loss展开实验。

* 更多的迭代轮数。蒸馏的baseline实验只迭代了120个epoch。实验发现，迭代轮数越多，蒸馏效果越好，最终我们迭代了360epoch，精度指标可以达到77.1%(`76.2%->77.1%`)。

* 无需数据集的真值标签，很容易扩展训练集。SSLD的loss在计算过程中，仅涉及到教师和学生模型对于相同图片的处理结果（经过softmax激活函数处理之后的soft label），因此即使图片数据不包含真值标签，也可以用来进行训练并提升模型性能。该蒸馏方案的无标签蒸馏策略也大大提升了学生模型的性能上限（`77.1%->78.5%`）。

* ImageNet1k蒸馏finetune。我们仅使用ImageNet1k数据，使用蒸馏方法对上述模型进行finetune，最终仍然可以获得0.4%的性能提升(`78.5%->78.9%`)。



## 2.2 数据选择


* SSLD蒸馏方案的一大特色就是无需使用图像的真值标签，因此可以任意扩展数据集的大小，考虑到计算资源的限制，我们在这里仅基于ImageNet22k数据集对蒸馏任务的训练集进行扩充。在SSLD蒸馏任务中，我们使用了`Top-k per class`的数据采样方案[3]。具体步骤如下。
    * 训练集去重。我们首先基于SIFT特征相似度匹配的方式对ImageNet22k数据集与ImageNet1k验证集进行去重，防止添加的ImageNet22k训练集中包含ImageNet1k验证集图像，最终去除了4511张相似图片。部分过滤的相似图片如下所示。

    ![](../../../images/distillation/22k_1k_val_compare_w_sift.png)

    * 大数据集soft label获取，对于去重后的ImageNet22k数据集，我们使用`ResNeXt101_32x16d_wsl`模型进行预测，得到每张图片的soft label。
    * Top-k数据选择，ImageNet1k数据共有1000类，对于每一类，找出属于该类并且得分最高的k张图片，最终得到一个数据量不超过`1000*k`的数据集（某些类上得到的图片数量可能少于k张）。
    * 将该数据集与ImageNet1k的训练集融合组成最终蒸馏模型所使用的数据集，数据量为500万。


# 三、实验

* PaddleClas的蒸馏策略为`大数据集训练+ImageNet1k蒸馏finetune`的策略。选择合适的教师模型，首先在挑选得到的500万数据集上进行训练，然后在ImageNet1k训练集上进行finetune，最终得到蒸馏后的学生模型。

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

针对从ImageNet22k挑选出的400万数据，融合imagenet1k训练集，组成共500万的训练集进行训练，具体地，在不同模型上的训练超参及效果如下。


|Student Model | num_epoch  | l2_ecay | batch size/gpu cards |  base lr | learning rate decay | top1 acc |
| - |:-: |:-: | :-: |:-: |:-: |:-: |
| MobileNetV1 | 360 | 3e-5 | 4096/8  | 1.6 | cosine_decay_warmup | 77.65% |
| MobileNetV2 | 360 | 1e-5 | 3072/8  | 0.54 | cosine_decay_warmup | 76.34% |
| MobileNetV3_large_x1_0 | 360 | 1e-5 |  5760/24 | 3.65625 | cosine_decay_warmup | 78.54% |
| MobileNetV3_small_x1_0 | 360 | 1e-5 |  5760/24 | 3.65625 | cosine_decay_warmup | 70.11% |
| ResNet50_vd | 360 | 7e-5 | 1024/32 | 0.4 | cosine_decay_warmup | 82.07% |
| ResNet101_vd | 360 | 7e-5 | 1024/32 | 0.4 | cosine_decay_warmup | 83.41% |
| Res2Net200_vd_26w_4s | 360 | 4e-5 | 1024/32 | 0.4 | cosine_decay_warmup | 84.82% |

## 3.3 ImageNet1k训练集finetune

对于在大数据集上训练的模型，其学习到的特征可能与ImageNet1k数据特征有偏，因此在这里使用ImageNet1k数据集对模型进行finetune。finetune的超参和finetune的精度收益如下。


|Student Model | num_epoch  | l2_ecay | batch size/gpu cards |  base lr | learning rate decay |  top1 acc |
| - |:-: |:-: | :-: |:-: |:-: |:-: |
| MobileNetV1 | 30 | 3e-5 | 4096/8 | 0.016 | cosine_decay_warmup | 77.89%  |
| MobileNetV2 | 30 | 1e-5 | 3072/8  | 0.0054 | cosine_decay_warmup | 76.73% |
| MobileNetV3_large_x1_0 | 30 | 1e-5 |  2048/8 | 0.008 | cosine_decay_warmup | 78.96% |
| MobileNetV3_small_x1_0 | 30 | 1e-5 |  6400/32 | 0.025 | cosine_decay_warmup | 71.28% |
| ResNet50_vd | 60 | 7e-5 | 1024/32 | 0.004 | cosine_decay_warmup | 82.39% |
| ResNet101_vd | 30 | 7e-5 | 1024/32 | 0.004 | cosine_decay_warmup | 83.73% |
| Res2Net200_vd_26w_4s | 360 | 4e-5 | 1024/32 | 0.004 | cosine_decay_warmup | 85.13% |


## 3.4 数据增广以及基于Fix策略的微调

* 基于前文所述的实验结论，我们在训练的过程中加入自动增广(AutoAugment)[4]，同时进一步减小了l2_decay(4e-5->2e-5)，最终ResNet50_vd经过SSLD蒸馏策略，在ImageNet1k上的精度可以达到82.99%，相比之前不加数据增广的蒸馏策略再次增加了0.6%。


* 对于图像分类任务，在测试的时候，测试尺度为训练尺度的1.15倍左右时，往往在不需要重新训练模型的情况下，模型的精度指标就可以进一步提升[5]，对于82.99%的ResNet50_vd在320x320的尺度下测试，精度可达83.7%，我们进一步使用Fix策略，即在320x320的尺度下进行训练，使用与预测时相同的数据预处理方法，同时固定除FC层以外的所有参数，最终在320x320的预测尺度下，精度可以达到**84.0%**。


## 3.4 实验过程中的一些问题

### 3.4.1 bn的计算方法

* 在预测过程中，batch norm的平均值与方差是通过加载预训练模型得到（设其模式为test mode）。在训练过程中，batch norm是通过统计当前batch的信息（设其模式为train mode），与历史保存信息进行滑动平均计算得到，在蒸馏任务中，我们发现通过train mode，即教师模型的bn实时变化的模式，去指导学生模型，比通过test mode蒸馏，得到的学生模型性能更好一些，下面是一组实验结果。因此我们在该蒸馏方案中，均使用train mode去得到教师模型的soft label。

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

# 四、蒸馏模型的应用

## 4.1 使用方法

* 中间层学习率调整。蒸馏得到的模型的中间层特征图更加精细化，因此将蒸馏模型预训练应用到其他任务中时，如果采取和之前相同的学习率，容易破坏中间层特征。而如果降低整体模型训练的学习率，则会带来训练收敛速度慢的问题。因此我们使用了中间层学习率调整的策略。具体地：
    * 针对ResNet50_vd，我们设置一个学习率倍数列表，res block之前的3个conv2d卷积参数具有统一的学习率倍数，4个res block的conv2d分别有一个学习率参数，共需设置5个学习率倍数的超参。在实验中发现。用于迁移学习finetune分类模型时，`[0.1,0.1,0.2,0.2,0.3]`的中间层学习率倍数设置在绝大多数的任务中都性能更好；而在目标检测任务中，`[0.05,0.05,0.05,0.1,0.15]`的中间层学习率倍数设置能够带来更大的精度收益。
    * 对于MoblileNetV3_large_1x0，由于其包含15个block，我们设置每3个block共享一个学习率倍数参数，因此需要共5个学习率倍数的参数，最终发现在分类和检测任务中，`[0.25,0.25,0.5,0.5,0.75]`的中间层学习率倍数能够带来更大的精度收益。


* 适当的l2 decay。不同分类模型在训练的时候一般都会根据模型设置不同的l2 decay，大模型为了防止过拟合，往往会设置更大的l2 decay，如ResNet50等模型，一般设置为`1e-4`；而如MobileNet系列模型，在训练时往往都会设置为`1e-5~4e-5`，防止模型过度欠拟合，在蒸馏时亦是如此。在将蒸馏模型应用到目标检测任务中时，我们发现也需要调节backbone甚至特定任务模型模型的l2 decay，和预训练蒸馏时的l2 decay尽可能保持一致。以Faster RCNN MobiletNetV3 FPN为例，我们发现仅修改该参数，在COCO2017数据集上就可以带来最多0.5%左右的精度(mAP)提升（默认Faster RCNN l2 decay为1e-4，我们修改为1e-5~4e-5均有0.3%~0.5%的提升）。


## 4.2 迁移学习finetune
* 为验证迁移学习的效果，我们在10个小的数据集上验证其效果。在这里为了保证实验的可对比性，我们均使用ImageNet1k数据集训练的标准预处理过程，对于蒸馏模型我们也添加了蒸馏模型中间层学习率的搜索。
* 对于ResNet50_vd，baseline为Top1 Acc 79.12%的预训练模型基于grid search搜索得到的最佳精度，对比实验则为基于该精度对预训练和中间层学习率进一步搜索得到的最佳精度。下面给出10个数据集上所有baseline和蒸馏模型的精度对比。


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


## 4.3 目标检测

我们基于两阶段目标检测Faster/Cascade RCNN模型验证蒸馏得到的预训练模型的效果。

* ResNet50_vd

设置训练与评测的尺度均为640x640，最终COCO上检测指标如下。

| Model | train/test scale | pretrain top1 acc | feature map lr | coco mAP |
|- |:-: |:-: | :-: | :-: |
| Faster RCNN R50_vd FPN | 640/640 | 79.12% | [1.0,1.0,1.0,1.0,1.0] | 34.8% |
| Faster RCNN R50_vd FPN | 640/640 | 79.12% | [0.05,0.05,0.1,0.1,0.15] | 34.3% |
| Faster RCNN R50_vd FPN | 640/640 | 82.18% | [0.05,0.05,0.1,0.1,0.15] | 36.3% |

在这里可以看出，对于未蒸馏模型，过度调整中间层学习率反而降低最终检测模型的性能指标。基于该蒸馏模型，我们也提供了领先的服务端实用目标检测方案，详细的配置与训练代码均已开源，可以参考[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_enhance)。


# 五、SSLD实战

本节将基于ImageNet-1K的数据集详细介绍SSLD蒸馏实验，如果想快速体验此方法，可以参考[**30分钟玩转PaddleClas**](../../tutorials/quick_start.md)中基于Flowers102的SSLD蒸馏实验。

## 5.1 参数配置

实战部分提供了SSLD蒸馏的示例，在`ppcls/modeling/architectures/distillation_models.py`中提供了`ResNeXt101_32x16d_wsl`蒸馏`ResNet50_vd`与`ResNet50_vd_ssld`蒸馏`MobileNetV3_large_x1_0`的示例，`configs/Distillation`里分别提供了二者的配置文件，用户可以在`tools/run.sh`里直接替换配置文件的路径即可使用。

### ResNeXt101_32x16d_wsl蒸馏ResNet50_vd

`ResNeXt101_32x16d_wsl`蒸馏`ResNet50_vd`的配置如下，其中`pretrained model`指定了`ResNeXt101_32x16d_wsl`（教师模型）的预训练模型的路径，该路径也可以同时指定教师模型与学生模型的预训练模型的路径，用户只需要同时传入二者预训练的路径即可（配置中的注释部分）。

```yaml
ARCHITECTURE:
    name: 'ResNeXt101_32x16d_wsl_distill_ResNet50_vd'
pretrained_model: "./pretrained/ResNeXt101_32x16d_wsl_pretrained/"
# pretrained_model:
#     - "./pretrained/ResNeXt101_32x16d_wsl_pretrained/"
#     - "./pretrained/ResNet50_vd_pretrained/"
use_distillation: True
```

### ResNet50_vd_ssld蒸馏MobileNetV3_large_x1_0

类似于`ResNeXt101_32x16d_wsl`蒸馏`ResNet50_vd`，`ResNet50_vd_ssld`蒸馏`MobileNetV3_large_x1_0`的配置如下:

```yaml
ARCHITECTURE:
    name: 'ResNet50_vd_distill_MobileNetV3_large_x1_0'
pretrained_model: "./pretrained/ResNet50_vd_ssld_pretrained/"
# pretrained_model:
#     - "./pretrained/ResNet50_vd_ssld_pretrained/"
#     - "./pretrained/ResNet50_vd_pretrained/"
use_distillation: True
```

## 5.2 启动命令

当用户配置完训练环境后，类似于训练其他分类任务，只需要将`tools/run.sh`中的配置文件替换成为相应的蒸馏配置文件即可。

其中`run.sh`中的内容如下：

```bash
export PYTHONPATH=path_to_PaddleClas:$PYTHONPATH

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    --log_dir=R50_vd_distill_MV3_large_x1_0 \
    tools/train.py \
        -c ./configs/Distillation/R50_vd_distill_MV3_large_x1_0.yaml
```

运行`run.sh`：

```bash
sh tools/run.sh
```

## 5.3 注意事项

* 用户在使用SSLD蒸馏之前，首先需要在目标数据集上训练一个教师模型，该教师模型用于指导学生模型在该数据集上的训练。

* 在用户使用SSLD蒸馏的时候需要将配置文件中的`use_distillation`设置为`True`，另外由于学生模型学习带有知识信息的soft-label，所以需要关掉label_smoothing选项，即将`ls_epsilon`中的值设置在[0,1]之外。

* 如果学生模型没有加载预训练模型，训练的其他超参数可以参考该学生模型在ImageNet-1k上训练的超参数，如果学生模型加载了预训练模型，学习率可以调整到原来的1/10或者1/100。

* 在SSLD蒸馏的过程中，学生模型只学习soft-label导致训练目标变的更加复杂，建议可以适当的调小`l2_decay`的值来获得更高的验证集准确率。

* 若用户准备添加无标签的训练数据，只需要将新的训练数据放置在原本训练数据的路径下，生成新的数据list即可，另外，新生成的数据list需要将无标签的数据添加伪标签（只是为了统一读数据）。


> 如果您觉得此文档对您有帮助，欢迎star我们的项目：[https://github.com/PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas)


# 参考文献

[1] Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network[J]. arXiv preprint arXiv:1503.02531, 2015.

[2] Bagherinezhad H, Horton M, Rastegari M, et al. Label refinery: Improving imagenet classification through label progression[J]. arXiv preprint arXiv:1805.02641, 2018.

[3] Yalniz I Z, Jégou H, Chen K, et al. Billion-scale semi-supervised learning for image classification[J]. arXiv preprint arXiv:1905.00546, 2019.

[4] Cubuk E D, Zoph B, Mane D, et al. Autoaugment: Learning augmentation strategies from data[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2019: 113-123.

[5] Touvron H, Vedaldi A, Douze M, et al. Fixing the train-test resolution discrepancy[C]//Advances in Neural Information Processing Systems. 2019: 8250-8260.
