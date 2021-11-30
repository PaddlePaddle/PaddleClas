# 更新日志

- 2021.11.1 发布[PP-ShiTu技术报告](https://arxiv.org/pdf/2111.00775.pdf)，新增饮料识别demo。
- 2021.10.23 发布轻量级图像识别系统PP-ShiTu，CPU上0.2s即可完成在10w+库的图像识别。[点击这里](../quick_start/quick_start_recognition.md)立即体验。
- 2021.09.17 发布PP-LCNet系列超轻量骨干网络模型, 在Intel CPU上，单张图像预测速度约5ms，ImageNet-1K数据集上Top1识别准确率达到80.82%，超越ResNet152的模型效果。PP-LCNet的介绍可以参考[论文](https://arxiv.org/pdf/2109.15099.pdf), 或者[PP-LCNet模型介绍](../models/PP-LCNet.md)，相关指标和预训练权重可以从 [这里](../algorithm_introduction/ImageNet_models.md)下载。
- 2021.08.11 更新7个[FAQ](../faq_series/faq_2021_s2.md)。
- 2021.06.29 添加Swin-transformer系列模型，ImageNet1k数据集上Top1 acc最高精度可达87.2%；支持训练预测评估与whl包部署，预训练模型可以从[这里](../models/models_intro.md)下载。
- 2021.06.22,23,24 PaddleClas官方研发团队带来技术深入解读三日直播课。课程回放：[https://aistudio.baidu.com/aistudio/course/introduce/24519](https://aistudio.baidu.com/aistudio/course/introduce/24519)
- 2021.06.16 PaddleClas v2.2版本升级，集成Metric learning，向量检索等组件。新增商品识别、动漫人物识别、车辆识别和logo识别等4个图像识别应用。新增LeViT、Twins、TNT、DLA、HarDNet、RedNet系列30个预训练模型。
- 2021.04.15
   - 添加`MixNet_L`和`ReXNet_3_0`系列模型，在ImageNet-1k上`MixNet` 模型Top1 Acc可达78.6%，`ReXNet`模型可达82.09%
- 2021.01.27
   * 添加ViT与DeiT模型，在ImageNet上，ViT模型Top-1 Acc可达81.05%，DeiT模型可达85.5%。
- 2021.01.08
    * 添加whl包及其使用说明，直接安装paddleclas whl包，即可快速完成模型预测。
- 2020.12.16
    * 添加对cpp预测的tensorRT支持，预测加速更明显。
- 2020.12.06
    * 添加SE_HRNet_W64_C_ssld模型，在ImageNet上Top-1 Acc可达0.8475。
- 2020.11.23
    * 添加GhostNet_x1_3_ssld模型，在ImageNet上Top-1 Acc可达0.7938。
- 2020.11.09
    * 添加InceptionV3结构和模型，在ImageNet上Top-1 Acc可达0.791。
- 2020.10.20
    * 添加Res2Net50_vd_26w_4s_ssld模型，在ImageNet上Top-1 Acc可达0.831；添加Res2Net101_vd_26w_4s_ssld模型，在ImageNet上Top-1 Acc可达0.839。
- 2020.10.12
    * 添加Paddle-Lite demo。
- 2020.10.10
    * 添加cpp inference demo。
    * 添加FAQ30问。
- 2020.09.17
    * 添加HRNet_W48_C_ssld模型，在ImageNet上Top-1 Acc可达0.836；添加ResNet34_vd_ssld模型，在ImageNet上Top-1 Acc可达0.797。

* 2020.09.07
    * 添加HRNet_W18_C_ssld模型，在ImageNet上Top-1 Acc可达0.81162；添加MobileNetV3_small_x0_35_ssld模型，在ImageNet上Top-1 Acc可达0.5555。

* 2020.07.14
    * 添加Res2Net200_vd_26w_4s_ssld模型，在ImageNet上Top-1 Acc可达85.13%。
    * 添加Fix_ResNet50_vd_ssld_v2模型，，在ImageNet上Top-1 Acc可达84.0%。

* 2020.06.17
    * 添加英文文档。

* 2020.06.12
    * 添加对windows和CPU环境的训练与评估支持。

* 2020.05.17
    * 添加混合精度训练。

* 2020.05.09
    * 添加Paddle Serving使用文档。
    * 添加Paddle-Lite使用文档。
    * 添加T4 GPU的FP32/FP16预测速度benchmark。

* 2020.04.10:
    * 第一次提交。
