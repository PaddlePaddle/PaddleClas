# PaddleClas 模型在昆仑芯上的部署方案-FastDeploy

## 1. 说明  
PaddleClas支持通过FastDeploy在昆仑芯上部署图像分类模型.

## 2. 模型版本说明

- [PaddleClas Release/2.4](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4)

目前FastDeploy支持如下模型的部署

- [PP-LCNet系列模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNet.md)
- [PP-LCNetV2系列模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md)
- [EfficientNet系列模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/EfficientNet_and_ResNeXt101_wsl.md)
- [GhostNet系列模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Mobile.md)
- [MobileNet系列模型(包含v1,v2,v3)](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Mobile.md)
- [ShuffleNet系列模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Mobile.md)
- [SqueezeNet系列模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Others.md)
- [Inception系列模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/Inception.md)
- [PP-HGNet系列模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-HGNet.md)
- [ResNet系列模型（包含vd系列）](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/ResNet_and_vd.md)

### 2.1 准备PaddleClas部署模型

PaddleClas模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/inference_deployment/export_model.md#2-%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA)  

注意：PaddleClas导出的模型仅包含`inference.pdmodel`和`inference.pdiparams`两个文件，但为了满足部署的需求，同时也需准备其提供的通用[inference_cls.yaml](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/deploy/configs/inference_cls.yaml)文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息，开发者可直接下载此文件使用。但需根据自己的需求修改yaml文件中的配置参数，具体可比照PaddleClas模型训练[config](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4/ppcls/configs/ImageNet)中的infer部分的配置信息进行修改。


### 2.2 下载预训练模型

为了方便开发者的测试，下面提供了PaddleClas导出的部分模型（含inference_cls.yaml文件），开发者可直接下载使用。

| 模型                                                               | 参数文件大小    |输入Shape |  Top1 | Top5 |
|:---------------------------------------------------------------- |:----- |:----- | :----- | :----- |
| [PPLCNet_x1_0](https://bj.bcebos.com/paddlehub/fastdeploy/PPLCNet_x1_0_infer.tgz) | 12MB | 224x224 |71.32% | 90.03% |
| [PPLCNetV2_base](https://bj.bcebos.com/paddlehub/fastdeploy/PPLCNetV2_base_infer.tgz)  | 26MB  | 224x224 |77.04% | 93.27% |
| [EfficientNetB7](https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB7_infer.tgz) |  255MB | 600x600 | 84.3% | 96.9% |
| [EfficientNetB0_small](https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB0_small_infer.tgz)|  18MB | 224x224 | 75.8% | 75.8% |
| [GhostNet_x1_3_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/GhostNet_x1_3_ssld_infer.tgz) |  29MB | 224x224 | 75.7% | 92.5% |
| [GhostNet_x0_5](https://bj.bcebos.com/paddlehub/fastdeploy/GhostNet_x0_5_infer.tgz) |  10MB | 224x224 | 66.8% | 86.9% |
| [MobileNetV1_x0_25](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_x0_25_infer.tgz) |  1.9MB | 224x224 | 51.4% | 75.5% |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_ssld_infer.tgz) |  17MB | 224x224 | 77.9% | 93.9% |
| [MobileNetV2_x0_25](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV2_x0_25_infer.tgz) |  5.9MB | 224x224 | 53.2% | 76.5% |
| [MobileNetV2_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV2_ssld_infer.tgz) |  14MB | 224x224 | 76.74% | 93.39% |
| [MobileNetV3_small_x0_35_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_small_x0_35_ssld_infer.tgz) |  6.4MB | 224x224 | 55.55% | 77.71% |
| [MobileNetV3_large_x1_0_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_large_x1_0_ssld_infer.tgz) |  22MB | 224x224 | 78.96% | 94.48% |
| [ShuffleNetV2_x0_25](https://bj.bcebos.com/paddlehub/fastdeploy/ShuffleNetV2_x0_25_infer.tgz) |  2.4MB | 224x224 | 49.9% | 73.79% |
| [ShuffleNetV2_x2_0](https://bj.bcebos.com/paddlehub/fastdeploy/ShuffleNetV2_x2_0_infer.tgz) |  29MB | 224x224 | 73.15% | 91.2% |
| [SqueezeNet1_1](https://bj.bcebos.com/paddlehub/fastdeploy/SqueezeNet1_1_infer.tgz) |  4.8MB | 224x224 | 60.1% | 81.9% |
| [InceptionV3](https://bj.bcebos.com/paddlehub/fastdeploy/InceptionV3_infer.tgz) |  92MB | 299x299 | 79.14% | 94.59% |
| [PPHGNet_tiny_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_tiny_ssld_infer.tgz) |  57MB | 224x224 | 81.95% | 96.12% |
| [PPHGNet_base_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_base_ssld_infer.tgz) |  274MB | 224x224 | 85.0% | 97.35% |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz) |  98MB | 224x224 | 79.12% | 94.44% |


## 3. 详细部署的部署示例  
- [Python部署](python)
- [C++部署](cpp)
