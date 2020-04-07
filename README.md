# PaddleCLS

## 简介
PaddleCLS的目的是为工业界和学术界提供一个图像分类任务相关的百宝箱，特色如下：
- 模型库：提供17种分类网络结构以及调参技巧，118个分类预训练模型以及性能评估

- 高阶使用：提供高精度的实用模型蒸馏方案（准确率82.39%的ResNet50_vd和78.9%的MobileNetV3）、8种数据增广方法的复现和验证

- 应用拓展：提供在常见视觉任务的特色方案，包括图像分类领域的迁移学习、通用目标检测、自然场景文字检测和识别等

- 实用工具：提供便于工业应用部署的实用工具，包括TensorRT预测、移动端预测、INT8量化和多机训练

- 赛事支持：助力多个视觉全球挑战赛取得领先成绩，包括2018年Kaggle Open Images V4图像目标检测挑战赛冠军、2019年Kaggle地标检索挑战赛亚军等
    
## 模型库

<div align="center">
    <img src="docs/images/models/main_fps_top1.png" width="600">
</div>

基于ImageNet1k分类数据集，PaddleCLS提供ResNet、ResNet_vd、EfficientNet、Res2Net、HRNet、MobileNetV3等17种主流分类网络结构的简单介绍，论文指标复现配置，以及在复现过程中的调参技巧。与此同时，PaddleCLS也提供了118个图像分类预训练模型，并且基于TensorRT评估了所有模型的GPU预测时间，以及在骁龙855（SD855）上评估了移动端模型的CPU预测时间和存储大小。

上图展示了一些适合服务器端应用的模型，使用V100 GPU，FP16和TensorRT预测一个batch的时间，其中batch_size=32，图中ResNet50_vd_ssld，是采用PaddleCLS提供的SSLD蒸馏方法训练的模型。不同模型的Floaps和Params、FP16和FP32的预测时间以及不同batch_size的预测时间持续更新中。

<div align="center">
<img
src="docs/images/models/mobile_arm_top1.png" width="1000">
</div>

上图展示了一些适合移动端应用的模型，在SD855上预测一张图像的CPU时间以及模型的存储大小。图中MV3_large_x1_0_ssld（M是MobileNet的简称），MV3_small_x1_0_ssld和MV1_ssld，是采用PaddleCLS提供的SSLD蒸馏方法训练的模型。MV3_large_x1_0_ssld_int8是进一步进行INT8量化的模型。不同模型的Floaps和Params、以及更多的GPU预测时间持续更新中。

- ToDo  
  - EfficientLite 论文指标复现和性能评估
  - GhostNet 论文指标复现和性能评估
  - RegNet 论文指标复现和性能评估

## 高阶使用
除了提供丰富的分类网络结构和预训练模型，PaddleCLS也提供了一系列有助于图像分类任务效果和效率提升的算法或工具。
- 模型蒸馏

模型蒸馏是指使用教师模型(teacher model)去指导学生模型(student model)学习特定任务，保证小模型在参数量不变的情况下，得到比较大的效果提升，甚至获得与大模型相似的精度指标。PaddleCLS提供了一种简单的半监督标签模型蒸馏方案（SSLD，Simple Semi-supervised Label Distillation），使用该方案大幅提升了ResNet50_vd、MobileNetV1和MobileNetV3在ImageNet数据集上分类效果。该蒸馏方案的框架图和蒸馏模型效果如下图所示，详细的蒸馏方法介绍以及使用持续更新中。

<div align="center">
<img
src="docs/images/distillation/ppcls_distillation_v1.png" width="600">
</div>

<div align="center">
<img
src="docs/images/distillation/distillation_perform.png" width="500">
</div>

- 数据增广

在图像分类任务中，图像数据的增广是一种常用的正则化方法，可以有效提升图像分类的效果，尤其对于数据量不足或者模型网络较深的场景。PaddleCLS提供了最新的8种数据增广算法的复现和在统一实验环境下效果评估，如下图所示。每种数据增广方法的详细介绍、对比的实验环境以及使用持续更新中。

<div align="center">
<img
src="docs/images/image_aug/main_image_aug.png" width="600">
</div> 

- ToDo
  - 更多的优化器支持和效果验证
  - 支持模型可解释性工具

## 应用拓展
效果更优的图像分类网络结构和预训练模型往往有助于提升其他视觉任务的效果，PaddleCLS提供了一系列在常见视觉任务中的特色方案。

- 图像分类的迁移学习

在实际应用中，由于训练数据的匮乏，往往将ImageNet1K数据集训练的分类模型作为预训练模型，进行图像分类的迁移学习。为了进一步助力实际问题的解决，PaddleCLS计划提供百度自研的基于10万种类别，4千多万的有标签数据训练的预训练模型，同时给出不同的超参搜索方法。该部分内容还在持续更新中。

- 通用目标检测

近年来，学术界和工业界广泛关注图像中目标检测任务。PaddleCLS基于82.39%的ResNet50_vd的预训练模型，结合PaddleDetection中丰富的检测算子，提供了一种面向服务器端应用的目标检测方案，PSS-DET (Practical Server Side Detection)，在COCO目标检测数据集上，当V100单卡预测速度为61FPS时，mAP是41.6%，预测速度为20FPS时，mAP是47.8%。详细的网络配置和训练代码，请参看<a href="https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_server_side_det" rel="nofollow"> PaddleDetection中的相关内容</a>。更多的PaddleCLS在目标检测中的特色应用，还在持续更新中。

<div align="center">
<img
src="docs/images/det/pssdet.png" width="500">
</div>

- ToDo
  - PaddleCLS在OCR任务中的特色应用
  - PaddleCLS在人脸检测和识别中的特色应用

## 实用工具
为了便于工业应用部署，PaddleCLS也提供了一些实用工具，持续更新中。

- TensorRT预测
- 移动端预测
- INT8量化
- 多机训练

## 赛事支持
PaddleCLS的建设源于百度实际视觉业务应用的淬炼和视觉前沿能力的探索，助力多个视觉重点赛事取得领先成绩，并且持续推进更多的前沿视觉问题的解决和落地应用。

- 2018年Kaggle Open Images V4图像目标检测挑战赛冠军
- 2019年Kaggle Open Images V5图像目标检测挑战赛亚军
- 2019年Kaggle地标检索挑战赛亚军
- 2019年Kaggle地标识别挑战赛亚军
- 首届多媒体信息识别技术竞赛中印刷文本OCR、人脸识别和地标识别三项任务A级证书

## 许可证书
本项目的发布受<a href="/PaddlePaddle/PaddleCLS/LICENSE">Apache 2.0 license</a>许可认证。

## 版本更新

## 如何贡献代码
我们非常欢迎你可以为PaddleCLS提供代码，也十分感谢你的反馈。
