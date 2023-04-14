# PaddleClas CPU-GPU C++部署示例

本目录下提供`infer.cc`快速完成PaddleClas系列模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

## 1. 说明  
PaddleClas支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署图像分类模型.

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库.

## 3. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleClas模型列表](../README.md)中下载所需模型.

## 4. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.0以上(x.x.x>=1.0.0)

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/cpu-gpu/cpp

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/cpu-gpu/cpp

mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg


# 在CPU上使用Paddle Inference推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 0
# 在CPU上使用OenVINO推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 1
# 在CPU上使用ONNX Runtime推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 2
# 在CPU上使用Paddle Lite推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 3
# 在GPU上使用Paddle Inference推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 4
# 在GPU上使用Paddle TensorRT推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 5
# 在GPU上使用ONNX Runtime推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 6
# 在GPU上使用Nvidia TensorRT推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 7
```
运行完成后返回结果如下所示
```bash
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)


## 5. 部署示例选项说明  
在我们使用`infer_demo`时, 输入了3个参数, 分别为分类模型, 预测图片, 与最后一位的数字选项.
现在下表将解释最后一位数字选项的含义.
|数字选项|含义|
|:---:|:---:|
|0| 在CPU上使用Paddle Inference推理 |
|1| 在CPU上使用OenVINO推理 |
|2| 在CPU上使用ONNX Runtime推理 |
|3| 在CPU上使用Paddle Lite推理 |
|4| 在GPU上使用Paddle Inference推理 |
|5| 在GPU上使用Paddle TensorRT推理 |
|6| 在GPU上使用ONNX Runtime推理 |
|7| 在GPU上使用Nvidia TensorRT推理 |

- 关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

## 6. 更多指南
- [PaddleClas系列 C++ API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1classification.html)
- [PaddleClas Python部署](../python)
- [PaddleClas C 部署](../c)
- [PaddleClas C# 部署](../csharp)

## 7. 常见问题
- PaddleClas能在FastDeploy支持的多种后端上推理,支持情况如下表所示, 如何切换后端, 详见文档[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

|硬件类型|支持的后端|
|:---:|:---:|
|X86 CPU| Paddle Inference, ONNX Runtime, OpenVINO |
|ARM CPU| Paddle Lite |
|飞腾 CPU| ONNX Runtime |
|NVIDIA GPU| Paddle Inference, ONNX Runtime, TensorRT |

- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)
