
# 飞桨训推一体认证

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleOCR中所有模型的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

<div align="center">
    <img src="docs/guide.png" width="1000">
</div>

## 2. 汇总信息

打通情况汇总如下，已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：包括模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度。
- 模型压缩：包括裁剪、离线/在线量化、蒸馏。
- 其他预测部署：包括Paddle Inference C++预测、Paddle Serving部署、Paddle-Lite部署等。

更详细的mkldnn、Tensorrt等预测加速相关功能的支持情况可以查看各测试工具的[更多教程](#more)。
| 算法论文 | 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 | 模型压缩 |  其他预测部署  |
| :--- | :--- |  :----:  | :--------: |  :----  |   :----  |   :----  |
| ResNet     |ResNet50_vd | 分类  | 支持 | 多机多卡 <br> 混合精度 | FPGM裁剪 <br> PACT量化|  |
| MobileNetV3     |MobileNetV3_large_x1_0 | 分类  | 支持 | 多机多卡 <br> 混合精度 | FPGM裁剪 <br> PACT量化|  |
| PPLCNet     |PPLCNet_x2_5 | 分类  | 支持 | 多机多卡 <br> 混合精度 | FPGM裁剪 <br> PACT量化|  |

## 3. 一键测试工具使用
### 目录介绍
```
./test_tipc/
├── common_func.sh                      #test_*.sh会调用到的公共函数
├── config     # 配置文件目录
│   ├── MobileNetV3_large_x1_0   # MobileNetV3_large_x1_0模型测试配置文件目录
│   │   ├── train_infer_python.txt                                        #基础训练预测配置文件
│   │   ├── train_linux_gpu_fleet_amp_infer_python_linux_gpu_cpu.txt      #多机多卡训练预测配置文件
│   │   └── train_linux_gpu_normal_amp_infer_python_linux_gpu_cpu.txt     #混合精度训练预测配置文件
│   └── ResNet50_vd              # ResNet50_vd模型测试配置文件目录
│       ├── train_infer_python.txt                                        #基础训练预测配置文件
│       ├── train_linux_gpu_fleet_amp_infer_python_linux_gpu_cpu.txt      #多机多卡训练预测配置文件
│       └── train_linux_gpu_normal_amp_infer_python_linux_gpu_cpu.txt     #混合精度训练预测配置文件
|   ......
├── docs
│   ├── guide.png
│   └── test.png
├── prepare.sh                          # 完成test_*.sh运行所需要的数据和模型下载
├── README.md                           # 使用文档
├── results                             # 预先保存的预测结果，用于和实际预测结果进行精读比对
└── test_train_inference_python.sh      # 测试python训练预测的主程序
```

### 测试流程
使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程如下：
<div align="center">
    <img src="docs/test.png" width="800">
</div>

1. 运行prepare.sh准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_*.sh`，产出log，由log可以看到不同配置是否运行成功；
3. 用`compare_results.py`对比log中的预测结果和预存在results目录下的结果，判断预测精度是否符合预期（在误差范围内）。

其中，有4个测试主程序，功能如下：
- `test_train_inference_python.sh`：测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。
- `test_inference_cpp.sh`：测试基于C++的模型推理。
- `test_serving.sh`：测试基于Paddle Serving的服务化部署功能。
- `test_lite.sh`：测试基于Paddle-Lite的端侧预测部署功能。

<a name="more"></a>
#### 更多教程
各功能测试中涉及混合精度、裁剪、量化等训练相关，及mkldnn、Tensorrt等多种预测相关参数配置，请点击下方相应链接了解更多细节和使用教程：  
[test_train_inference_python 使用](docs/test_train_inference_python.md)  
[test_inference_cpp 使用](docs/test_inference_cpp.md)  
[test_serving 使用](docs/test_serving.md)  
[test_lite 使用](docs/test_lite.md)
