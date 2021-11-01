简体中文 | [English](README_en.md)

# PaddleClas

## 简介

飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，助力使用者训练出更好的视觉模型和应用落地。

**近期更新**
- 2021.10.31 发布[PP-ShiTu技术报告](./docs/PP_ShiTu.pdf)，优化文档，新增饮料识别demo
- 2021.10.23 发布PP-ShiTu图像识别系统，新增轻量级检测、特征提取模型，速度提升800%，新增DeepHash模块，检索模块切换为faiss，支持PaddleServing和PaddleSlim
- 2021.09.17 增加PaddleClas自研PP-LCNet系列模型, 这些模型在Intel CPU上有较强的竞争力。PP-LCNet的介绍可以参考[论文](https://arxiv.org/pdf/2109.15099.pdf)或者[PP-LCNet模型介绍](docs/zh_CN/models/PP-LCNet.md)，相关指标和预训练权重可以从 [这里](docs/zh_CN/ImageNet_models_cn.md)下载。
- 2021.08.11 更新7个[FAQ](docs/zh_CN/faq_series/faq_2021_s2.md)。
- 2021.06.29 添加Swin-transformer系列模型，ImageNet1k数据集上Top1 acc最高精度可达87.2%；支持训练预测评估与whl包部署，预训练模型可以从[这里](docs/zh_CN/models/models_intro.md)下载。
- 2021.06.22,23,24 PaddleClas官方研发团队带来技术深入解读三日直播课。课程回放：[https://aistudio.baidu.com/aistudio/course/introduce/24519](https://aistudio.baidu.com/aistudio/course/introduce/24519)
- 2021.06.16 PaddleClas v2.2版本升级，集成Metric learning，向量检索等组件。新增商品识别、动漫人物识别、车辆识别和logo识别等4个图像识别应用。新增LeViT、Twins、TNT、DLA、HarDNet、RedNet系列30个预训练模型。
- [more](./docs/zh_CN/others/update_history.md)

<a name="图像识别系统介绍"></a>
## PP-ShiTu轻量级图像识别系统

<div align="center">
<img src="./docs/images/recognition.gif"  width = "400" />
</div>

- 图像识别系统：集成了目标检测、特征学习、图像检索等模块，一套模型适用多个场景，下载即用。

- 轻量级检测、特征提取模型：CPU预测速度大幅提升，部分场景速度较上一版模型提升800%，精度打平。

<div align="center">
<img src="./docs/images/structure.jpg"  width = "800" />
</div>

- 整个图像识别系统分为三步：
  1. 通过一个目标检测模型，检测图像物体候选区域
  2. 对每个候选区域进行特征提取
  3. 与检索库中图像进行特征匹配，提取识别结果。 
  - 对于新的未知类别，无需重新训练模型，只需要在检索库补入该类别图像，重新建立检索库，就可以识别该类别。

- 更详细的内容请参见我们的[技术报告](./docs/PP_ShiTu.pdf)

## PaddleClas特性

- 丰富的预训练模型库：提供了35个系列共164个ImageNet预训练模型，其中6个精选系列模型支持结构快速修改。

- 全面易用的特征学习组件：集成arcmargin, triplet loss等12度量学习方法，通过配置文件即可随意组合切换。

- SSLD知识蒸馏：14个分类预训练模型，精度普遍提升3%以上；其中ResNet50_vd模型在ImageNet-1k数据集上的Top-1精度达到了84.0%，
Res2Net200_vd预训练模型Top-1精度高达85.1%。

- 数据增广：支持AutoAugment、Cutout、Cutmix等8种数据增广算法详细介绍、代码复现和在统一实验环境下的效果评估。

## 欢迎加入技术交流群

* 您可以扫描下面的微信群二维码， 加入PaddleClas 微信交流群。获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。

<div align="center">
<img src="./docs/images/wx_group.png"  width = "200" />
</div>

## 快速体验
图像识别快速体验：[点击这里](./docs/zh_CN/quick_start/quick_start_recognition.md)

## 文档教程

- [快速安装](./docs/zh_CN/installation/install_paddleclas.md)
- [图像识别快速体验](./docs/zh_CN/quick_start/quick_start_recognition.md)
- [PP-ShiTu图像识别系统介绍](#图像识别系统介绍)
- [识别效果展示](#识别效果展示)
- 图像分类快速体验
    - [尝鲜版](./docs/zh_CN/quick_start/quick_start_classification_new_user.md)
    - [进阶版](./docs/zh_CN/quick_start/quick_start_classification_professional.md)
- 算法介绍
    - [骨干网络和预训练模型库](./docs/zh_CN/algorithm_introduction/ImageNet_models.md)
    - [主体检测](./docs/zh_CN/image_recognition_pipeline/mainbody_detection.md)
    - [图像分类](./docs/zh_CN/algorithm_introduction/image_classification.md)
    - [特征学习](./docs/zh_CN/algorithm_introduction/metric_learning.md)
    - [向量检索](./deploy/vector_search/README.md)
- 模型训练/评估
    - [图像分类任务](./docs/zh_CN/models_training/classification.md)
    - [特征学习任务](./docs/zh_CN/models_training/recognition.md)
- 模型预测
    - [基于Python预测引擎预测推理](./docs/zh_CN/inference_deployment/python_deploy.md)
    - [基于C++预测引擎预测推理](./deploy/cpp/readme.md)(当前只支持图像分类任务，图像识别更新中)
- 模型部署
    - [Paddle Serving服务化部署(推荐)](./docs/zh_CN/inference_deployment/paddle_serving_deploy.md)
    - [Hub serving服务化部署](./docs/zh_CN/inference_deployment/paddle_hub_serving_deploy.md)
    - [端侧部署](./deploy/lite/readme.md)
    - [whl包预测](./docs/zh_CN/inference_deployment/whl_deploy.md)
- 高阶使用
    - [知识蒸馏](./docs/zh_CN/advanced_tutorials/knowledge_distillation.md)
    - [模型量化](./docs/zh_CN/advanced_tutorials/model_prune_quantization.md)
    - [数据增广](./docs/zh_CN/advanced_tutorials/DataAugmentation.md)
- FAQ
    - [图像识别任务FAQ](docs/zh_CN/faq_series/faq_2021_s2.md)
    - [图像分类任务FAQ](docs/zh_CN/faq_series/faq.md)
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)

<a name="识别效果展示"></a>
## 更多效果展示 [more](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.2/docs/images/recognition/more_demo_images)
- 瓶装饮料识别
<div align="center">
<img src="docs/images/drink_demo.gif">
</div>

- 商品识别
<div align="center">
<img src="https://user-images.githubusercontent.com/18028216/122769644-51604f80-d2d7-11eb-8290-c53b12a5c1f6.gif"  width = "400" />
</div>

- 动漫人物识别
<div align="center">
<img src="https://user-images.githubusercontent.com/18028216/122769746-6b019700-d2d7-11eb-86df-f1d710999ba6.gif"  width = "400" />
</div>

- logo识别
<div align="center">
<img src="https://user-images.githubusercontent.com/18028216/122769837-7fde2a80-d2d7-11eb-9b69-04140e9d785f.gif"  width = "400" />
</div>


- 车辆识别
<div align="center">
<img src="https://user-images.githubusercontent.com/18028216/122769916-8ec4dd00-d2d7-11eb-8c60-42d89e25030c.gif"  width = "400" />
</div>


<a name="许可证书"></a>

## 许可证书
本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleCLS/blob/master/LICENSE">Apache 2.0 license</a>许可认证。


<a name="贡献代码"></a>
## 贡献代码
我们非常欢迎你为PaddleClas贡献代码，也十分感谢你的反馈。
如果想为PaddleCLas贡献代码，可以参考[贡献指南](./docs/zh_CN/advanced_tutorials/how_to_contribute.md)。


- 非常感谢[nblib](https://github.com/nblib)修正了PaddleClas中RandErasing的数据增广配置文件。
- 非常感谢[chenpy228](https://github.com/chenpy228)修正了PaddleClas文档中的部分错别字。
- 非常感谢[jm12138](https://github.com/jm12138)为PaddleClas添加ViT，DeiT系列模型和RepVGG系列模型。
- 非常感谢[FutureSI](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/76563)对PaddleClas代码的解析与总结。

我们非常欢迎你为PaddleClas贡献代码，也十分感谢你的反馈。
