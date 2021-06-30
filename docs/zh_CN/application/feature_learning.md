# 特征学习

此部分主要是针对特征学习的训练模式进行说明，即`RecModel`的训练模式。主要是为了支持车辆识别（车辆细分类、ReID）、Logo识别、动漫人物识别、商品识别等特征学习的应用。与在`ImageNet`上训练普通的分类网络不同的是，此特征学习部分，主要有以下特征

- 支持对`backbone`的输出进行截断，即支持提取任意中间层的特征信息
- 支持在`backbone`的feature输出层后，添加可配置的网络层，即`Neck`部分
- 支持`ArcFace Loss`等`metric learning` 相关loss函数，提升特征学习能力

## 1 整体流程

![](../../images/recognition/rec_pipeline.png)

特征学习的整体结构如上图所示，主要包括：数据增强、Backbone的设置、Neck、Metric Learning等几大部分。其中`Neck`部分为自由添加的网络层，如添加的embedding层等，当然也可以不用此模块。训练时，利用`Metric Learning`部分的Loss对模型进行优化。预测时，一般来说，默认以`Neck`部分的输出作为特征输出。

针对不同的应用，可以根据需要，对每一部分自由选择。每一部分的具体配置，如数据增强、Backbone、Neck、Metric Learning相关Loss等设置，详见具体应用：[车辆识别](./vehicle_recognition.md)、[Logo识别](./logo_recognition.md)、[动漫人物识别](./cartoon_character_recognition.md)、[商品识别](./product_recognition.md)

## 2 配置文件说明

配置文件说明详见[yaml配置文件说明文档](../tutorials/config.md)。其中模型结构配置，详见文档中**识别模型结构配置**部分。

## 3 预训练模型

以下为各应用在不同数据集下的预训练模型

- 车辆细分类：[CompCars](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/vehicle_cls_ResNet50_CompCars_v1.1_pretrained.pdparams)
- 车辆ReID：[VERI-Wild](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/vehicle_reid_ResNet50_VERIWild_v1.0_pretrained.pdparams)
- 动漫人物识别：[iCartoon](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/cartoon_rec_ResNet50_iCartoon_v1.0_pretrained.pdparams)
- Logo识别：[Logo3K](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/logo_rec_ResNet50_Logo3K_v1.0_pretrained.pdparams)
- 商品识别： [Inshop](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/product_ResNet50_vd_Inshop_pretrained_v1.0.pdparams)、[Aliproduct](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained.pdparams) 
