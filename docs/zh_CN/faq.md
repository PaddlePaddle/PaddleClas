# FAQ

## 写在前面

* 我们收集整理了开源以来在issues和用户群中的常见问题并且给出了简要解答，旨在为图像分类的开发者提供一些参考，也希望帮助大家少走一些弯路。

* 图像分类领域大佬众多，模型和论文更新速度也很快，本文档回答主要依赖有限的项目实践，难免挂一漏万，如有遗漏和不足，也希望有识之士帮忙补充和修正，万分感谢。


## PaddleClas常见问题汇总

* [图像分类30个问题](#图像分类30个问题)
    * [基础知识](#基础知识)
    * [模型训练相关](#模型训练相关)
    * [数据相关](#数据相关)
    * [模型推理与预测相关](#模型推理与预测相关)
* [PaddleClas使用问题](#PaddleClas使用问题)


<a name="图像分类30个问题"></a>
## 图像分类30个问题

<a name="基础知识"></a>
### 基础知识

>>
* Q: 图像分类领域常用的分类指标有几种
* A:
    * 对于单个标签的图像分类问题（仅包含1个类别与背景），评估指标主要有Accuracy，Precision，Recall，F-score等，令TP(True Positive)表示将正类预测为正类，FP(False Positive)表示将负类预测为正类，TN(True Negative)表示将负类预测为负类，FN(False Negative)表示将正类预测为负类。那么Accuracy=(TP + TN) / NUM，Precision=TP /(TP + FP)，Recall=TP /(TP + FN)。
    * 对于类别数大于1的图像分类问题，评估指标主要有Accuary和Class-wise Accuracy，Accuary表示所有类别预测正确的图像数量占总图像数量的百分比；Class-wise Accuracy是对每个类别的图像计算Accuracy，然后再对所有类别的Accuracy取平均得到。

>>
* Q: 怎样根据自己的任务选择合适的模型进行训练？
* A: 如果希望在服务器部署，或者希望精度尽可能地高，对模型存储大小或者预测速度的要求不是很高，那么推荐使用ResNet_vd、Res2Net_vd、DenseNet、Xception等适合于服务器端的系列模型；如果希望在移动端侧部署，则推荐使用MobileNetV3、GhostNet等适合于移动端的系列模型。同时，我们推荐在选择模型的时候可以参考[模型库](https://github.com/PaddlePaddle/PaddleClas/tree/master/docs/zh_CN/models)中的速度-精度指标图。

>>
* Q: 如何进行参数初始化，什么样的初始化可以加快模型收敛？
* A: 众所周知，参数的初始化可以影响模型的最终性能。一般来说，如果目标数据集不是很大，建议使用ImageNet-1k训练得到的预训练模型进行初始化。如果是自己手动设计的网络或者暂时没有基于ImageNet-1k训练得到的预训练权重，可以使用Xavier初始化或者MSRA初始化，其中Xavier初始化是针对Sigmoid函数提出的，对RELU函数不太友好，网络越深，各层输入的方差越小，网络越难训练，所以当神经网络中使用较多RELU激活函数时，推荐使用MSRA初始化。

>>
* Q: 针对深度神经网络参数冗余的问题，目前有哪些比较好的解决办法？
* A: 目前有几种主要的方法对模型进行压缩，减少模型参数冗余的问题，如剪枝、量化、知识蒸馏等。模型剪枝指的是将权重矩阵中相对不重要的权值剔除，然后再重新对网络进行微调；模型量化指的是一种将浮点计算转成低比特定点计算的技术，如8比特、4比特等，可以有效的降低模型计算强度、参数大小和内存消耗。知识蒸馏是指使用教师模型(teacher model)去指导学生模型(student model)学习特定任务，保证小模型在参数量不变的情况下，性能有较大的提升，甚至获得与大模型相似的精度指标。

>>
* Q: 怎样在其他任务，如目标检测、图像分割、关键点检测等任务中选择比较合适的分类模型作为骨干网络？
* A: 在不考虑速度的情况下，在大部分的任务中，推荐使用精度更高的预训练模型和骨干网络，PaddleClas中开源了一系列的SSLD知识蒸馏预训练模型，如ResNet50_vd_ssld, Res2Net200_vd_26w_4s_ssld等，在模型精度和速度方面都是非常有优势的，推荐大家使用。对于一些特定的任务，如图像分割或者关键点检测等任务，对图像分辨率的要求比较高，那么更推荐使用HRNet等能够同时兼顾网络深度和分辨率的神经网络模型，PaddleClas也提供了HRNet_W18_C_ssld、HRNet_W48_C_ssld等精度非常高的HRNet SSLD蒸馏系列预训练模型，大家可以使用这些精度更高的预训练模型与骨干网络，提升自己在其他任务上的模型精度。

>>
* Q: 注意力机制是什么？目前有哪些比较常用的注意力机制方法？
* A: 注意力机制（Attention Mechanism）源于对人类视觉的研究。将注意力机制用在计算机视觉任务上，可以有效捕捉图片中有用的区域，从而提升整体网络性能。目前比较常用的有[SE block](https://arxiv.org/abs/1709.01507)、[SK-block](https://arxiv.org/abs/1903.06586)、[Non-local block](https://arxiv.org/abs/1711.07971)、[GC block](https://arxiv.org/abs/1904.11492)、[CBAM](https://arxiv.org/abs/1807.06521)等，核心思想就是去学习特征图在不同区域或者不同通道中的重要性，从而让网络更加注意显著性的区域。

<a name="模型训练相关"></a>
### 模型训练相关

>>
* Q: 使用深度卷积网络做图像分类如果训练一个拥有1000万个类的模型会碰到什么问题？
* A: 因为FC层参数很多，内存/显存/模型的存储占用都会大幅增大；模型收敛速度也会变慢一些。建议在这种情况下，再最后的FC层前加一层维度较小的FC，这样可以大幅减少模型的存储大小。

>>
* Q: 训练过程中，如果模型收敛效果很差，可能的原因有哪些呢？
* A: 主要有以下几个可以排查的地方：（1）应该检查数据标注，确保训练集和验证集的数据标注没有问题。（2）可以试着调整一下学习率（初期可以以10倍为单位进行调节），过大（训练震荡）或者过小（收敛太慢）的学习率都可能导致收敛效果差。（3）数据量太大，选择的模型太小，难以学习所有数据的特征。（4）可以看下数据预处理的过程中是否使用了归一化，如果没有使用归一化操作，收敛速度可能会比较慢。（5）如果数据量比较小，可以试着加载PaddleClas中提供的基于ImageNet-1k数据集的预训练模型，这可以大大提升训练收敛速度。（6）数据集存在长尾问题，可以参考[数据长尾问题解决方案](#jump)。

>>
* Q: 训练图像分类任务时，该怎么选择合适的优化器？
* A: 优化器的目的是为了让损失函数尽可能的小，从而找到合适的参数来完成某项任务。目前业界主要用到的优化器有SGD、RMSProp、Adam、AdaDelt等，其中由于带momentum的SGD优化器广泛应用于学术界和工业界，所以我们发布的模型也大都使用该优化器来实现损失函数的梯度下降。带momentum的SGD优化器有两个劣势，其一是收敛速度慢，其二是初始学习率的设置需要依靠大量的经验，然而如果初始学习率设置得当并且迭代轮数充足，该优化器也会在众多的优化器中脱颖而出，使得其在验证集上获得更高的准确率。一些自适应学习率的优化器如Adam、RMSProp等，收敛速度往往比较快，但是最终的收敛精度会稍差一些。如果追求更快的收敛速度，我们推荐使用这些自适应学习率的优化器，如果追求更高的收敛精度，我们推荐使用带momentum的SGD优化器。

>>
* Q: 当前主流的学习率下降策略有哪些？一般需要怎么选择呢？
* A: 学习率是通过损失函数的梯度调整网络权重的超参数的速度。学习率越低，损失函数的变化速度就越慢。虽然使用低学习率可以确保不会错过任何局部极小值，但也意味着将花费更长的时间来进行收敛，特别是在被困在高原区域的情况下。在整个训练过程中，我们不能使用同样的学习率来更新权重，否则无法到达最优点，所以需要在训练过程中调整学习率的大小。在训练初始阶段，由于权重处于随机初始化的状态，损失函数相对容易进行梯度下降，所以可以设置一个较大的学习率。在训练后期，由于权重参数已经接近最优值，较大的学习率无法进一步寻找最优值，所以需要设置一个较小的学习率。在训练整个过程中，很多研究者使用的学习率下降方式是piecewise_decay，即阶梯式下降学习率，如在ResNet50标准的训练中，我们设置的初始学习率是0.1，每30epoch学习率下降到原来的1/10，一共迭代120epoch。除了piecewise_decay，很多研究者也提出了学习率的其他下降方式，如polynomial_decay（多项式下降）、exponential_decay（指数下降）,cosine_decay（余弦下降）等，其中cosine_decay无需调整超参数，鲁棒性也比较高，所以成为现在提高模型精度首选的学习率下降方式。Cosine_decay和piecewise_decay的学习率变化曲线如下图所示，容易观察到，在整个训练过程中，cosine_decay都保持着较大的学习率，所以其收敛较为缓慢，但是最终的收敛效果较peicewise_decay更好一些。
![](../images/models/lr_decay.jpeg)
>>
* Q: Warmup学习率策略是什么？一般用在什么样的场景中？
* A: Warmup策略顾名思义就是让学习率先预热一下，在训练初期我们不直接使用最大的学习率，而是用一个逐渐增大的学习率去训练网络，当学习率增大到最高点时，再使用学习率下降策略中提到的学习率下降方式衰减学习率的值。如果使用较大的batch_size训练神经网络时，我们建议您使用warmup策略。实验表明，在batch_size较大时，warmup可以稳定提升模型的精度。在训练MobileNetV3等batch_size较大的实验中，我们默认将warmup中的epoch设置为5，即先用5epoch将学习率从0增加到最大值，再去做相应的学习率衰减。

>>
* Q: 什么是`batch size`？在模型训练中，怎么选择合适的`batch size`？
* A: `batch size`是训练神经网络中的一个重要的超参数，该值决定了一次将多少数据送入神经网络参与训练。论文[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)，当`batch size`的值与学习率的值呈线性关系时，收敛精度几乎不受影响。在训练ImageNet数据时，大部分的神经网络选择的初始学习率为0.1，`batch size`是256，所以根据实际的模型大小和显存情况，可以将学习率设置为0.1*k,batch_size设置为256*k。在实际任务中，也可以将该设置作为初始参数，进一步调节学习率参数并获得更优的性能。
>>
* Q: weight_decay是什么？怎么选择合适的weight_decay呢？
* A: 过拟合是机器学习中常见的一个名词，简单理解即为模型在训练数据上表现很好，但在测试数据上表现较差，在卷积神经网络中，同样存在过拟合的问题，为了避免过拟合，很多正则方式被提出，其中，weight_decay是其中一个广泛使用的避免过拟合的方式。在使用SGD优化器时，weight_decay等价于在最终的损失函数后添加L2正则化，L2正则化使得网络的权重倾向于选择更小的值，最终整个网络中的参数值更趋向于0，模型的泛化性能相应提高。在各大深度学习框架的实现中，该值表达的含义是L2正则前的系数，在paddle框架中，该值的名称是l2_decay，所以以下都称其为l2_decay。该系数越大，表示加入的正则越强，模型越趋于欠拟合状态。在训练ImageNet的任务中，大多数的网络将该参数值设置为1e-4，在一些小的网络如MobileNet系列网络中，为了避免网络欠拟合，该值设置为1e-5~4e-5之间。当然，该值的设置也和具体的数据集有关系，当任务的数据集较大时，网络本身趋向于欠拟合状态，可以将该值适当减小，当任务的数据集较小时，网络本身趋向于过拟合状态，可以将该值适当增大。下表展示了MobileNetV1_x0_25在ImageNet-1k上使用不同l2_decay的精度情况。由于MobileNetV1_x0_25是一个比较小的网络，所以l2_decay过大会使网络趋向于欠拟合状态，所以在该网络中，相对1e-4，3e-5是更好的选择。

| 模型                | L2_decay | Train acc1/acc5 | Test acc1/acc5 |
|:--:|:--:|:--:|:--:|
| MobileNetV1_x0_25 | 1e-4     | 43.79%/67.61%   | 50.41%/74.70%  |
| MobileNetV1_x0_25 | 3e-5     | 47.38%/70.83%   | 51.45%/75.45%  |


>>
* Q: 标签平滑(label_smoothing)指的是什么？有什么效果呢？一般适用于什么样的场景中？
* A: Label_smoothing是深度学习中的一种正则化方法，其全称是 Label Smoothing Regularization(LSR)，即标签平滑正则化。在传统的分类任务计算损失函数时，是将真实的one hot标签与神经网络的输出做相应的交叉熵计算，而label_smoothing是将真实的one hot标签做一个标签平滑的处理，使得网络学习的标签不再是一个hard label，而是一个有概率值的soft label，其中在类别对应的位置的概率最大，其他位置概率是一个非常小的数。具体的计算方式参见论文[2]。在label_smoothing里，有一个epsilon的参数值，该值描述了将标签软化的程度，该值越大，经过label smoothing后的标签向量的标签概率值越小，标签越平滑，反之，标签越趋向于hard label，在训练ImageNet-1k的实验里通常将该值设置为0.1。
在训练ImageNet-1k的实验中，我们发现，ResNet50大小级别及其以上的模型在使用label_smooting后，精度有稳定的提升。下表展示了ResNet50_vd在使用label_smoothing前后的精度指标。同时，由于label_smoohing相当于一种正则方式，在相对较小的模型上，精度提升不明显甚至会有所下降，下表展示了ResNet18在ImageNet-1k上使用label_smoothing前后的精度指标。可以明显看到，在使用label_smoothing后，精度有所下降。

| 模型   | Use_label_smoothing | Test acc1 |
|:--:|:--:|:--:|
| ResNet50_vd | 0    | 77.9%  |
| ResNet50_vd | 1    | 78.4%  |
| ResNet18    | 0    | 71.0%  |
| ResNet18    | 1    | 70.8%  |


>>
* Q: 在训练的时候怎么通过训练集和验证集的准确率或者loss确定进一步的调优策略呢？
* A: 在训练网络的过程中，通常会打印每一个epoch的训练集准确率和验证集准确率，二者刻画了该模型在两个数据集上的表现。通常来说，训练集的准确率比验证集准确率微高或者二者相当是比较不错的状态。如果发现训练集的准确率比验证集高很多，说明在这个任务上已经过拟合，需要在训练过程中加入更多的正则，如增大l2_decay的值，加入更多的数据增广策略，加入label_smoothing策略等；如果发现训练集的准确率比验证集低一些，说明在这个任务上可能欠拟合，需要在训练过程中减弱正则效果，如减小l2_decay的值，减少数据增广方式，增大图片crop区域面积，减弱图片拉伸变换，去除label_smoothing等。

>>
* Q: 怎么使用已有的预训练模型提升自己的数据集的精度呢？
* A: 在现阶段计算机视觉领域中，加载预训练模型来训练自己的任务已成为普遍的做法，相比从随机初始化开始训练，加载预训练模型往往可以提升特定任务的精度。一般来说，业界广泛使用的预训练模型是通过训练128万张图片1000类的ImageNet-1k数据集得到的，该预训练模型的fc层权重是是一个k\*1000的矩阵，其中k是fc层以前的神经元数，在加载预训练权重时，无需加载fc层的权重。在学习率方面，如果您的任务训练的数据集特别小（如小于1千张），我们建议你使用较小的初始学习率，如0.001（batch_size:256,下同），以免较大的学习率破坏预训练权重。如果您的训练数据集规模相对较大（大于10万），我们建议你尝试更大的初始学习率，如0.01或者更大。

<a name="数据相关"></a>
### 数据相关

>>
* Q: 图像分类的数据预处理过程一般包括哪些步骤？
* A: 以在ImageNet-1k数据集上训练ResNet50为例，一张图片被输入进网络，主要有图像解码、随机裁剪、随机水平翻转、标准化、数据重排，组batch并送进网络这几个步骤。图像解码指的是将图片文件读入到内存中，随机裁剪指的是将读入的图像随机拉伸并裁剪到长宽均为224的图像，随机水平翻转指的是对裁剪后的图片以0.5的概率进行水平翻转，标准化指的是将图片每个通道的数据通过去均值实现中心化的处理，使得数据尽可能符合`N(0,1)`的正态分布，数据重排指的是将数据由`[224,224,3]`的格式变为`[3,224,224]`的格式，组batch指的是将多幅图像组成一个批数据，送进网络进行训练。

>>
* Q: 随机裁剪是怎么影响小模型训练的性能的？
* A: 在ImageNet-1k数据的标准预处理中，随机裁剪函数中定义了scale和ratio两个值，两个值分别确定了图片crop的大小和图片的拉伸程度，其中scale的默认取值范围是0.08-1(lower_scale-upper_scale),ratio的默认取值范围是3/4-4/3(lower_ratio-upper_ratio)。在非常小的网络训练中，此类数据增强会使得网络欠拟合，导致精度有所下降。为了提升网络的精度，可以使其数据增强变的更弱，即增大图片的crop区域或者减弱图片的拉伸变换程度。我们可以分别通过增大lower_scale的值或缩小lower_ratio与upper_scale的差距来实现更弱的图片变换。下表列出了使用不同lower_scale训练MobileNetV2_x0_25的精度，可以看到，增大图片的crop区域面积后训练精度和验证精度均有提升。

| 模型                | Scale取值范围 | Train_acc1/acc5 | Test_acc1/acc5 |
|:--:|:--:|:--:|:--:|
| MobileNetV2_x0_25 | [0.08,1]  | 50.36%/72.98%   | 52.35%/75.65%  |
| MobileNetV2_x0_25 | [0.2,1]   | 54.39%/77.08%   | 53.18%/76.14%  |


>>
* Q: 数据量不足的情况下，目前有哪些常见的数据增广方法来增加训练样本的丰富度呢？
* A: PaddleClas中将目前比较常见的数据增广方法分为了三大类，分别是图像变换类、图像裁剪类和图像混叠类，图像变换类主要包括AutoAugment和RandAugment，图像裁剪类主要包括CutOut、RandErasing、HideAndSeek和GridMask，图像混叠类主要包括Mixup和Cutmix，更详细的关于数据增广的介绍可以参考：[数据增广章节](./advanced_tutorials/image_augmentation/ImageAugment.md)。
>>
* Q: 对于遮挡情况比较常见的图像分类场景，该使用什么数据增广方法去提升模型的精度呢？
* A: 在训练的过程中可以尝试对训练集使用CutOut、RandErasing、HideAndSeek和GridMask等裁剪类数据增广方法，让模型也能够不止学习到显著区域，也能关注到非显著性区域，从而在遮挡的情况下，也能较好地完成识别任务。

>>
* Q: 对于色彩变换情况比较复杂的情况下，应该使用哪些数据增广方法提升模型精度呢？
* A: 可以考虑使用AutoAugment或者RandAugment的数据增广策略，这两种策略中都包括了锐化、直方图均衡化等丰富的颜色变换，可以让模型在训练的过程中对这些变换更加鲁棒。
>>
* Q: Mixup和Cutmix的工作原理是什么？为什么它们也是非常有效的数据增广方法？
* A: Mixup通过线性叠加两张图片生成新的图片，对应label也进行线性叠加用以训练，Cutmix则是从一幅图中随机裁剪出一个 感兴趣区域(ROI)，然后覆盖当前图像中对应的区域，label也按照图像面积比例进行线性叠加。它们其实也是生成了和训练集不同的样本和label并让网络去学习，从而扩充了样本的丰富度。
>>
* Q: 对于精度要求不是那么高的图像分类任务，大概需要准备多大的训练数据集呢？
* A: 训练数据的数量和需要解决问题的复杂度有关系。难度越大，精度要求越高，则数据集需求越大，而且一般情况实际中的训练数据越多效果越好。当然，一般情况下，在加载预训练模型的情况下，每个类别包括10-20张图像即可保证基本的分类效果；不加载预训练模型的情况下，每个类别需要至少包含100-200张图像以保证基本的分类效果。

>>
* Q: <span id="jump">对于长尾分布的数据集，目前有哪些比较常用的方法？</span>
* A: （1）可以对数据量比较少的类别进行重采样，增加其出现的概率；（2）可以修改loss，增加图像较少对应的类别的图片的loss权重；（3）可以借鉴迁移学习的方法，从常见类别中学习通用知识，然后迁移到少样本的类别中。

<a name="模型推理与预测相关"></a>
### 模型推理与预测相关

>>
* Q: 有时候图像中只有小部分区域是所关注的前景物体，直接拿原图来进行分类的话，识别效果很差，这种情况要怎么做呢？
* A: 可以在分类之前先加一个主体检测的模型，将前景物体检测出来之后再进行分类，可以大大提升最终的识别效果。如果不考虑时间成本，也可以使用multi-crop的方式对所有的预测做融合来决定最终的类别。
>>
* Q: 目前推荐的，模型预测方式有哪些？
* A: 在模型训练完成之后，推荐使用导出的固化模型（inference model），基于Paddle预测引擎进行预测，目前支持python inference与cpp inference。如果希望基于服务化部署预测模型，那么推荐使用HubServing的部署方式。
>>
* Q: 模型训练完成之后，有哪些比较合适的预测方法进一步提升模型精度呢？
* A: （1）可以使用更大的预测尺度，比如说训练的时候使用的是224，那么预测的时候可以考虑使用288或者320，这会直接带来0.5%左右的精度提升。（2）可以使用测试时增广的策略（Test Time Augmentation, TTA)，将测试集通过旋转、翻转、颜色变换等策略，创建多个副本，并分别预测，最后将所有的预测结果进行融合，这可以大大提升预测结果的精度和鲁棒性。（3）当然，也可以使用多模型融合的策略，将多个模型针对相同图片的预测结果进行融合。
>>
* Q: 多模型融合的时候，该怎么选择合适的模型进行融合呢？
* A: 在不考虑预测速度的情况下，建议选择精度尽量高的模型；同时建议选择不同结构或者系列的模型进行融合，比如在精度相似的情况下，ResNet50_vd与Xception65的模型融合结果往往比ResNet50_vd与ResNet101_vd的模型融合结果要好一些。

>>
* Q: 使用固定的模型进行预测时有哪些比较常用的加速方法？
* A: （1）使用性能更优的GPU进行预测；（2）增大预测的batch size；（3）使用TenorRT以及FP16半精度浮点数等方法进行预测。


<a name="PaddleClas使用问题"></a>
## PaddleClas使用问题

>>
* Q: 多卡评估时，为什么每张卡输出的精度指标不相同？
* A: 目前PaddleClas基于fleet api使用多卡，在多卡评估时，每张卡都是单独读取各自part的数据，不同卡中计算的图片是不同的，因此最终指标也会有微量差异，如果希望得到准确的评估指标，可以使用单卡评估。

>>
* Q: 在配置文件的`TRAIN`字段中配置了`mix`的参数，为什么`mixup`的数据增广预处理没有生效呢？
* A: 使用mixup时，数据预处理部分与模型输入部分均需要修改，因此还需要在配置文件中显式地配置`use_mix: True`，才能使得`mixup`生效。


>>
* Q: 评估和预测时，已经指定了预训练模型所在文件夹的地址，但是仍然无法导入参数，这么为什么呢？
* A: 加载预训练模型时，需要指定预训练模型的前缀，例如预训练模型参数所在的文件夹为`output/ResNet50_vd/19`，预训练模型参数的名称为`output/ResNet50_vd/19/ppcls.pdparams`，则`pretrained_model`参数需要指定为`output/ResNet50_vd/19/ppcls`，PaddleClas会自动补齐`.pdparams`的后缀。


>>
* Q: 在评测`EfficientNetB0_small`模型时，为什么最终的精度始终比官网的低0.3%左右？
* A: `EfficientNet`系列的网络在进行resize的时候，是使用`cubic插值方式`(resize参数的interpolation值设置为2)，而其他模型默认情况下为None，因此在训练和评估的时候需要显式地指定resize的interpolation值。具体地，可以参考以下配置中预处理过程中ResizeImage的参数。
```
VALID:
    batch_size: 16
    num_workers: 4
    file_list: "./dataset/ILSVRC2012/val_list.txt"
    data_dir: "./dataset/ILSVRC2012/"
    shuffle_seed: 0
    transforms:
        - DecodeImage:
            to_rgb: True
            to_np: False
            channel_first: False
        - ResizeImage:
            resize_short: 256
            interpolation: 2
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
```

>>
* Q: 如果想将保存的`pdparams`模型参数文件转换为早期版本(Paddle1.7.0之前)的零碎文件(每个文件均为一个单独的模型参数)，该怎么实现呢？
* A: 可以首先导入`pdparams`模型，之后使用`fluid.io.save_vars`函数将模型保存为零散的碎文件。示例代码如下，最终所有零散文件会被保存在`path_to_save_var`目录下。
```
fluid.load(
        program=infer_prog, model_path=args.pretrained_model, executor=exe)
state = fluid.io.load_program_state(args.pretrained_model)
def exists(var):
    return var.name in state
fluid.io.save_vars(exe, "./path_to_save_var", infer_prog, predicate=exists)
```

>>
* Q: python2下，使用visualdl的时候，报出以下错误，`TypeError: __init__() missing 1 required positional argument: 'sync_cycle'`，这是为什么呢？
* A: 目前visualdl仅支持在python3下运行，visualdl需要是2.0以上的版本，如果visualdl版本不对的话，可以通过以下方式进行安装：`pip3 install visualdl==2.0.0b8  -i https://mirror.baidu.com/pypi/simple`

>>
* Q: 自己在测ResNet50_vd预测单张图片速度的时候发现比官网提供的速度benchmark慢了很多，而且CPU速度比GPU速度快很多，这个是为什么呢？
* A: 模型预测需要初始化，初始化的过程比较耗时，因此在统计预测速度的时候，需要批量跑一批图片，去除前若干张图片的预测耗时，再统计下平均的时间。GPU比CPU速度测试单张图片速度慢是因为GPU的初始化并CPU要慢很多。

>>
* Q: 在动态图中加载静态图预训练模型的时候，需要注意哪些问题？
* A: 在使用infer.py预测单张图片或者文件夹中的图片时，需要注意指定[infer.py](https://github.com/PaddlePaddle/PaddleClas/blob/53c5850df7c49a1bfcd8d989e6ccbea61f406a1d/tools/infer/infer.py#L40)中的`load_static_weights`为True，在finetune或者评估的时候需要添加`-o load_static_weights=True`的参数。
>>
* Q: 灰度图可以用于模型训练吗？
* A: 灰度图也可以用于模型训练，不过需要修改模型的输入shape为`[1, 224, 224]`，此外数据增广部分也需要注意适配一下。不过为了更好地使用PaddleClas代码的话，即使是灰度图，也建议调整为3通道的图片进行训练（RGB通道的像素值相等）。

>>
* Q: 怎么在windows上或者cpu上面模型训练呢？
* A: 可以参考[PaddleClas开始使用教程](https://github.com/PaddlePaddle/PaddleClas/blob/master/docs/zh_CN/tutorials/getting_started.md)，详细介绍了在Linux、Windows、CPU等环境中进行模型训练、评估与预测的教程。
>>
* Q: 怎样在模型训练的时候使用label smoothing呢？
* A: 可以在配置文件中设置label smoothing epsilon的值，`ls_epsilon=0.1`，表示设置该值为0.1，若该值为-1，则表示不使用label smoothing。
>>
* Q: PaddleClas提供的10W类图像分类预训练模型能否用于模型推断呢？
* A: 该10W类图像分类预训练模型没有提供fc全连接层的参数，无法用于模型推断，目前可以用于模型微调。
>>
* Q: 在使用`tools/infere/predict.py`进行模型预测的时候，报了这个问题:`Error: Pass tensorrt_subgraph_pass has not been registered`，这是为什么呢？
* A: 如果希望使用TensorRT进行模型预测推理的话，需要编译带TensorRT的PaddlePaddle，编译的时候参考以下的编译方式，其中`TENSORRT_ROOT`表示TensorRT的路径。
```
cmake  .. \
        -DWITH_CONTRIB=OFF \
        -DWITH_MKL=ON \
        -DWITH_MKLDNN=ON  \
        -DWITH_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_INFERENCE_API_TEST=OFF \
        -DON_INFER=ON \
        -DWITH_PYTHON=ON \
        -DPY_VERSION=2.7 \
        -DTENSORRT_ROOT=/usr/local/TensorRT6-cuda10.0-cudnn7/
make -j16
make inference_lib_dist
```
>>
* Q: 怎样在训练的时候使用自动混合精度(Automatic Mixed Precision, AMP)训练呢？
* A: 可以参考[ResNet50_fp16.yml](https://github.com/PaddlePaddle/PaddleClas/blob/master/configs/ResNet/ResNet50_fp16.yml)这个配置文件；具体地，如果希望自己的配置文件在模型训练的时候也支持自动混合精度，可以在配置文件中添加下面的配置信息。
```
use_fp16: True
amp_scale_loss: 128.0
use_dynamic_loss_scaling: True
```
