
# PaddleClas 图像分类模瑞芯微NPU部署方案-FastDeploy

## 1. 说明  
本示例基于RV1126来介绍如何使用FastDeploy部署PaddleClas量化模型，支持如下芯片的部署：  
- Rockchip RV1109
- Rockchip RV1126
- Rockchip RK1808

## 2. 使用预导出的模型列表  

FastDeploy提供预先量化好的模型进行部署. 更多模型, 欢迎用户参考[FastDeploy 一键模型自动化压缩工具](https://github.com/PaddlePaddle/FastDeploy/tree/develop/tools/common_tools/auto_compression) 来实现模型量化, 并完成部署.


| 模型            | 量化方式 |
|:---------------| :----- |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | 离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)  | 离线量化 |

## 3. 详细部署示例
在 RV1126 上只支持 C++ 的部署。
- [C++部署](cpp)
