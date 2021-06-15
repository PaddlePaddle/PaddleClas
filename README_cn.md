简体中文 | [English](README.md)

# PaddleClas

## 简介

飞桨图像分类套件PaddleClas是飞桨为工业界和学术界所准备的一个图像分类任务的工具集，助力使用者训练出更好的视觉模型和应用落地。

**近期更新**

- 2021.06.16 PaddleClas v2.2版本升级，集成Metric learning，向量检索等组件，新增4个图像识别应用。
- [more](./docs/zh_CN/update_history.md)


## 特性

- 完整的图像识别解决方案：集成了检测、特征学习、检索等模块，广泛适用于各类图像识别任务。
提供商品识别、车辆识别、logo识别和动漫人物识别等4个示例解决方案。

- 丰富的预训练模型库：提供了29个系列共134个ImageNet预训练模型，其中6个精选系列模型支持结构快速修改。

- 全面易用的特征学习组件：集成大量度量学习方法，通过配置文件即可随意组合切换。

- SSLD知识蒸馏：基于该方案蒸馏模型的识别准确率普遍提升3%以上。

- 数据增广：支持AutoAugment、Cutout、Cutmix等8种数据增广算法详细介绍、代码复现和在统一实验环境下的效果评估。

## 欢迎加入技术交流群

* 微信扫描下面左方二维码添加飞桨小姐姐的微信，添加成功后私信小姐姐暗号【分类】，即可收到微信群进群邀请。

<div align="center">
<img src="./docs/images/joinus.png"  width = "200" />
</div>

* 您也可以扫描下面的QQ群二维码， 加入PaddleClas QQ交流群。获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。

<div align="center">
<img src="./docs/images/qq_group.png"  width = "200" />
</div>


## 文档教程

- [快速安装](./docs/zh_CN/tutorials/install.md)
- 图像识别快速体验(若愚)
- 图像分类快速体验(崔程，基于30分钟入门版修改)
- 算法介绍
    - 图像识别系统] (胜禹)
    - [模型库介绍和预训练模型](./docs/zh_CN/models/models_intro.md)
    - [图像分类]
        - ImageNet分类任务(崔程,基于30分钟进阶版修改)
        - [多标签分类任务]()
    - [特征学习](水龙)
        - [商品识别]()
        - [车辆识别]()
        - [logo识别]()
        - [动漫人物识别]()
    - [向量检索]()
- 模型训练/评估
    - 图像分类任务(崔程，基于原有训练文档整理)
    - 特征学习任务（陆彬）
- 模型预测
    - [基于训练引擎预测推理](./docs/zh_CN/tutorials/getting_started.md)
    - [基于Python预测引擎预测推理](./docs/zh_CN/tutorials/getting_started.md)
    - [基于C++预测引擎预测推理](./deploy/cpp_infer/readme.md)
    - [服务化部署](./deploy/hubserving/readme.md)
    - [端侧部署](./deploy/lite/readme.md)
    - [whl包预测](./docs/zh_CN/whl.md)
    - [模型量化压缩](deploy/slim/quant/README.md)
- 高阶使用
    - [知识蒸馏](./docs/zh_CN/advanced_tutorials/distillation/distillation.md)
    - [模型量化](./docs/zh_CN/extension/paddle_quantization.md)
    - [数据增广](./docs/zh_CN/advanced_tutorials/image_augmentation/ImageAugment.md)
    - [代码解析与社区贡献指南](./docs/zh_CN/tutorials/quick_start_community.md)
- FAQ(暂停更新)
    - [图像分类任务FAQ]
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)


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
