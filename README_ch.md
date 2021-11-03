简体中文 | [English](README_en.md)

# PaddleClas

## 简介

飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，助力使用者训练出更好的视觉模型和应用落地。

**近期更新**

- 2021.11.1 发布[PP-ShiTu技术报告](https://arxiv.org/pdf/2111.00775.pdf)，新增饮料识别demo
- 2021.10.23 发布轻量级图像识别系统PP-ShiTu，CPU上0.2s即可完成在10w+库的图像识别。
[点击这里](./docs/zh_CN/quick_start/quick_start_recognition.md)立即体验
- 2021.09.17 发布PP-LCNet系列超轻量骨干网络模型, 在Intel CPU上，单张图像预测速度约5ms，ImageNet-1K数据集上Top1识别准确率达到80.82%，超越ResNet152的模型效果。PP-LCNet的介绍可以参考[论文](https://arxiv.org/pdf/2109.15099.pdf), 或者[PP-LCNet模型介绍](docs/zh_CN/models/PP-LCNet.md)，相关指标和预训练权重可以从 [这里](docs/zh_CN/algorithm_introduction/ImageNet_models.md)下载。
- [more](./docs/zh_CN/others/update_history.md)

## 特性

- PP-ShiTu轻量图像识别系统：集成了目标检测、特征学习、图像检索等模块，广泛适用于各类图像识别任务。cpu上0.2s即可完成在10w+库的图像识别。

- PP-LCNet轻量级CPU骨干网络：专门为CPU设备打造轻量级骨干网络，速度、精度均远超竞品。

- 丰富的预训练模型库：提供了36个系列共175个ImageNet预训练模型，其中7个精选系列模型支持结构快速修改。

- 全面易用的特征学习组件：集成arcmargin, triplet loss等12度量学习方法，通过配置文件即可随意组合切换。

- SSLD知识蒸馏：14个分类预训练模型，精度普遍提升3%以上；其中ResNet50_vd模型在ImageNet-1k数据集上的Top-1精度达到了84.0%，
Res2Net200_vd预训练模型Top-1精度高达85.1%。

<div align="center">
<img src="./docs/images/recognition.gif"  width = "400" />
</div>


## 欢迎加入技术交流群

* 您可以扫描下面的微信群二维码， 加入PaddleClas 微信交流群。获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。

<div align="center">
<img src="./docs/images/wx_group.png"  width = "200" />
</div>

## 快速体验

PP-ShiTu图像识别快速体验：[点击这里](./docs/zh_CN/quick_start/quick_start_recognition.md)

## 文档教程
- 安装说明
  - [安装Paddle](./docs/zh_CN/installation/install_paddle.md)
  - [安装PaddleClas](./docs/zh_CN/installation/install_paddleclas.md)
- 快速体验
  - [PP-ShiTu图像识别快速体验](./docs/zh_CN/quick_start/quick_start_recognition.md)
  - 图像分类快速体验
    - [尝鲜版](./docs/zh_CN/quick_start/quick_start_classification_new_user.md)
    - [进阶版](./docs/zh_CN/quick_start/quick_start_classification_professional.md)
- [PP-ShiTu图像识别系统介绍](#图像识别系统介绍)
    - [主体检测](./docs/zh_CN/image_recognition_pipeline/mainbody_detection.md)
    - [特征提取](./docs/zh_CN/image_recognition_pipeline/feature_extraction.md)
    - [向量检索](./docs/zh_CN/image_recognition_pipeline/vector_search.md)
- [骨干网络和预训练模型库](./docs/zh_CN/algorithm_introduction/ImageNet_models.md)
- 数据准备
  - [图像分类数据集介绍](./docs/zh_CN/data_preparation/classification_dataset.md)
  - [图像识别数据集介绍](./docs/zh_CN/data_preparation/recognition_dataset.md)
- 模型训练
    - [图像分类任务](./docs/zh_CN/models_training/classification.md)
    - [图像识别任务](./docs/zh_CN/models_training/recognition.md)
    - [训练参数调整策略](./docs/zh_CN/models_training/train_strategy.md)
    - [配置文件说明](./docs/zh_CN/models_training/config_description.md)
- 模型预测部署
    - [模型导出](./docs/zh_CN/inference_deployment/export_model.md)
    - Python/C++ 预测引擎
      - [基于Python预测引擎预测推理](./docs/zh_CN/inference_deployment/python_deploy.md)
      - [基于C++预测引擎预测推理](./docs/zh_CN/inference_deployment/cpp_deploy.md)(当前只支持图像分类任务，图像识别更新中)
    - 服务化部署
      - [Paddle Serving服务化部署(推荐)](./docs/zh_CN/inference_deployment/paddle_serving_deploy.md)
      - [Hub serving服务化部署](./docs/zh_CN/inference_deployment/paddle_hub_serving_deploy.md)
    - [端侧部署](./deploy/lite/readme.md)
    - [whl包预测](./docs/zh_CN/inference_deployment/whl_deploy.md)
- 算法介绍
    - [图像分类任务介绍](./docs/zh_CN/algorithm_introduction/image_classification.md)
    - [度量学习介绍](./docs/zh_CN/algorithm_introduction/metric_learning.md)
- 高阶使用
    - [数据增广](./docs/zh_CN/advanced_tutorials/DataAugmentation.md)
    - [模型量化](./docs/zh_CN/advanced_tutorials/model_prune_quantization.md)
    - [知识蒸馏](./docs/zh_CN/advanced_tutorials/knowledge_distillation.md)
    - [PaddleClas结构解析](./docs/zh_CN/advanced_tutorials/code_overview.md)
    - [社区贡献指南](./docs/zh_CN/advanced_tutorials/how_to_contribute.md)
- FAQ
    - [图像识别精选问题](docs/zh_CN/faq_series/faq_2021_s2.md)
    - [图像分类精选问题](docs/zh_CN/faq_series/faq_selected_30.md)
    - [图像分类FAQ第一季](docs/zh_CN/faq_series/faq_2020_s1.md)
    - [图像分类FAQ第二季](docs/zh_CN/faq_series/faq_2021_s1.md)
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)

<a name="图像识别系统介绍"></a>
## PP-ShiTu图像识别系统介绍

<div align="center">
<img src="./docs/images/structure.jpg"  width = "800" />
</div>

PP-ShiTu是一个实用的轻量级通用图像识别系统，主要由主体检测、特征学习和向量检索三个模块组成。该系统从骨干网络选择和调整、损失函数的选择、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型裁剪量化8个方面，采用多种策略，对各个模块的模型进行优化，最终得到在CPU上仅0.2s即可完成10w+库的图像识别的系统。更多细节请参考[PP-ShiTu技术方案](https://arxiv.org/pdf/2111.00775.pdf)。


<a name="识别效果展示"></a>
## PP-ShiTu图像识别系统效果展示
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
