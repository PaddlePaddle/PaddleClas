# PaddleClas 模型在晶晨NPU上的部署方案-FastDeploy

## 1. 说明  

晶晨A311D是一款先进的AI应用处理器。PaddleClas支持通过FastDeploy在A311D上基于Paddle-Lite部署相关PaddleClas模型。**注意**：需要注意的是，芯原（verisilicon）作为 IP 设计厂商，本身并不提供实体SoC产品，而是授权其 IP 给芯片厂商，如：晶晨（Amlogic），瑞芯微（Rockchip）等。因此本文是适用于被芯原授权了 NPU IP 的芯片产品。只要芯片产品没有大副修改芯原的底层库，则该芯片就可以使用本文档作为 Paddle Lite 推理部署的参考和教程。在本文中，晶晨 SoC 中的 NPU 和 瑞芯微 SoC 中的 NPU 统称为芯原 NPU。目前支持如下芯片的部署：
- Amlogic A311D
- Amlogic C308X
- Amlogic S905D3

本示例基于晶晨A311D来介绍如何使用FastDeploy部署PaddleClas的量化模型。

## 2. 使用预导出的模型列表  

FastDeploy提供预先量化好的模型进行部署. 更多模型, 欢迎用户参考[FastDeploy 一键模型自动化压缩工具](https://github.com/PaddlePaddle/FastDeploy/tree/develop/tools/common_tools/auto_compression) 来实现模型量化, 并完成部署.


| 模型            | 量化方式 |
|:---------------| :----- |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | 离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)  | 离线量化 |

## 3. 详细部署示例

目前，A311D上只支持C++的部署。
- [C++部署](cpp)
