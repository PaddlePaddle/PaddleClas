简体中文 | [English](README_en.md)

# PaddleClas

## 简介

飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，助力使用者训练出更好的视觉模型和应用落地。

**近期更新**

- 2021.06.22,23,24 PaddleOCR官方研发团队带来技术深入解读三日直播课，6月22日、23日、24日晚上20:30，[直播地址](https://live.bilibili.com/21689802)
- 2021.06.16 PaddleClas v2.2版本升级，集成Metric learning，向量检索等组件。新增商品识别、动漫人物识别、车辆识别和logo识别等4个图像识别应用。新增LeViT、Twins、TNT、DLA、HarDNet、RedNet系列30个预训练模型。
- 2021.05.14 添加`SwinTransformer` 系列模型。
- 2021.04.15 添加`MixNet_L`和`ReXNet_3_0`系列模型。 
 
[more](./docs/zh_CN/update_history.md)

## 特性

- 实用的图像识别系统：集成了检测、特征学习、检索等模块，广泛适用于各类图像识别任务。
提供商品识别、车辆识别、logo识别和动漫人物识别等4个示例。

- 丰富的预训练模型库：提供了35个系列共164个ImageNet预训练模型，其中6个精选系列模型支持结构快速修改。

- 全面易用的特征学习组件：集成arcmargin, triplet loss等12度量学习方法，通过配置文件即可随意组合切换。

- SSLD知识蒸馏：14个分类预训练模型，精度普遍提升3%以上；其中ResNet50_vd模型在ImageNet-1k数据集上的Top-1精度达到了84.0%，
Res2Net200_vd预训练模型Top-1精度高达85.1%。

- 数据增广：支持AutoAugment、Cutout、Cutmix等8种数据增广算法详细介绍、代码复现和在统一实验环境下的效果评估。

## 图像识别系统效果展示
<div align="center">
<img src="./docs/images/recognition.gif"  width = "400" />
</div>

## 欢迎加入技术交流群

* 您也可以扫描下面的微信群二维码， 加入PaddleClas 微信交流群。获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。

<div align="center">
<img src="./docs/images/wx_group.png"  width = "200" />
</div>

## 快速体验
图像识别快速体验：[点击这里](./docs/zh_CN/tutorials/quick_start_recognition.md)

## 文档教程

- [快速安装](./docs/zh_CN/tutorials/install.md)
- [图像识别快速体验](./docs/zh_CN/tutorials/quick_start_recognition.md)
- 算法介绍（更新中）
    - [骨干网络和预训练模型库](./docs/zh_CN/ImageNet_models_cn.md)
    - [主体检测](./docs/zh_CN/application/object_detection.md)
    - 图像分类
        - [ImageNet分类任务](./docs/zh_CN/tutorials/quick_start_professional.md)
    - 特征学习
        - [商品识别](./docs/zh_CN/application/product_recognition.md)
        - [车辆识别](./docs/zh_CN/application/vehicle_reid.md)
        - [logo识别](./docs/zh_CN/application/logo_recognition.md)
        - [动漫人物识别](./docs/zh_CN/application/cartoon_character_recognition.md)
    - [向量检索](./deploy/vector_search/README.md)
- 模型训练/评估
    - [图像分类任务](./docs/zh_CN/tutorials/getting_started.md)
    - [特征学习任务](./docs/zh_CN/application/feature_learning.md)
- 模型预测（当前只支持图像分类任务，图像识别更新中）
    - [基于Python预测引擎预测推理](./docs/zh_CN/tutorials/getting_started.md)
    - [基于C++预测引擎预测推理](./deploy/cpp_infer/readme.md)
    - [服务化部署](./deploy/hubserving/readme.md)
    - [端侧部署](./deploy/lite/readme.md)
    - [whl包预测](./docs/zh_CN/whl.md)
- 高阶使用
    - [知识蒸馏](./docs/zh_CN/advanced_tutorials/distillation/distillation.md)
    - [模型量化](./docs/zh_CN/extension/paddle_quantization.md)
    - [数据增广](./docs/zh_CN/advanced_tutorials/image_augmentation/ImageAugment.md)
- FAQ(暂停更新)
    - [图像分类任务FAQ](docs/zh_CN/faq.md)
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)


## 图像识别系统介绍

<a name="图像识别系统介绍"></a>
<div align="center">
<img src="./docs/images/structure.png"  width = "400" />
</div>

整个图像识别系统分为三步：（1）通过一个目标检测模型，检测图像物体候选区域（2）对每个候选区域进行特征提取（3）与检索库中图像进行特征匹配，提取识别结果。

对于新的未知类别，无需重新训练模型，只需要在检索库补入该类别图像，重新建立检索库，就可以识别该类别。

<a name="许可证书"></a>

## 许可证书
本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleCLS/blob/master/LICENSE">Apache 2.0 license</a>许可认证。


<a name="贡献代码"></a>
## 贡献代码
我们非常欢迎你为PaddleClas贡献代码，也十分感谢你的反馈。

- 非常感谢[nblib](https://github.com/nblib)修正了PaddleClas中RandErasing的数据增广配置文件。
- 非常感谢[chenpy228](https://github.com/chenpy228)修正了PaddleClas文档中的部分错别字。
- 非常感谢[jm12138](https://github.com/jm12138)为PaddleClas添加ViT，DeiT系列模型和RepVGG系列模型。
- 非常感谢[FutureSI](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/76563)对PaddleClas代码的解析与总结。

我们非常欢迎你为PaddleClas贡献代码，也十分感谢你的反馈。
