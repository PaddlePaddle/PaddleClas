# PaddleClas CPU-GPU C#部署示例

本目录下提供`infer.cs`来调用C# API快速完成PaddleClas模型在CPU/GPU上部署的示例。

## 1. 说明  
PaddleClas支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署图像分类模型.

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库. 在本目录执行如下命令即可在Windows完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

## 3. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleClas模型列表](../README.md)中下载所需模型.

## 4. 部署示例

### 4.1 下载C#包管理程序nuget客户端
> https://dist.nuget.org/win-x86-commandline/v6.4.0/nuget.exe

下载完成后将该程序添加到环境变量**PATH**中

### 4.2 下载模型文件和测试图片
> https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz # (下载后解压缩)
> https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

### 4.3 编译示例代码

本文档编译的示例代码的编译工具依赖VS 2019，**Windows打开x64 Native Tools Command Prompt for VS 2019命令工具**，通过如下命令开始编译

```shell
## 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy\examples\vision\classification\paddleclas\cpu-gpu\csharp

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd D:\PaddleClas\deploy\fastdeploy\cpu-gpu\csharp

mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DFASTDEPLOY_INSTALL_DIR=D:\fastdeploy-win-x64-gpu-x.x.x -DCUDA_DIRECTORY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"

nuget restore
msbuild infer_demo.sln /m:4 /p:Configuration=Release /p:Platform=x64
```

关于使用Visual Studio 2019创建sln工程，或者CMake工程等方式编译的更详细信息，可参考如下文档
- [在 Windows 使用 FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/cn/faq/use_sdk_on_windows.md)
- [FastDeploy C++库在Windows上的多种使用方式](https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/cn/faq/use_sdk_on_windows_build.md)

## 4.4 运行可执行程序

注意Windows上运行时，需要将FastDeploy依赖的库拷贝至可执行程序所在目录, 或者配置环境变量。FastDeploy提供了工具帮助我们快速将所有依赖库拷贝至可执行程序所在目录,通过如下命令将所有依赖的dll文件拷贝至可执行程序所在的目录(可能生成的可执行文件在Release下还有一层目录，这里假设生成的可执行文件在Release处)
```shell
cd D:\Download\fastdeploy-win-x64-gpu-x.x.x

fastdeploy_init.bat install %cd% D:\PaddleClas\deploy\fastdeploy\cpu-gpu\csharp\build\Release
```

将dll拷贝到当前路径后，准备好模型和图片，使用如下命令运行可执行程序即可
```shell
cd Release
# CPU推理
infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 0
# GPU推理
infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 1
```

## 5. PaddleClas C#接口简介

下面提供了PaddleClas的C# API简介

- 如果用户想要更换部署后端或进行其他定制化操作, 请查看[C# Runtime API](https://github.com/PaddlePaddle/FastDeploy/blob/develop/csharp/fastdeploy/runtime_option.cs).
- 更多 PaddleClas C# API 请查看 [C# PaddleClas API](https://github.com/PaddlePaddle/FastDeploy/blob/develop/csharp/fastdeploy/vision/classification/ppcls/model.cs)

### 模型

```c#
fastdeploy.vision.classification.PaddleClasModel(
        string model_file,
        string params_file,
        string config_file,
        fastdeploy.RuntimeOption runtime_option = null,
        fastdeploy.ModelFormat model_format = ModelFormat.PADDLE)
```

> PaddleClasModel模型加载和初始化。

> **参数**

>> * **model_file**(str): 模型文件路径
>> * **params_file**(str): 参数文件路径
>> * **config_file**(str): 配置文件路径，即PaddleClas导出的部署yaml文件
>> * **runtime_option**(RuntimeOption): 后端推理配置，默认为null，即采用默认配置
>> * **model_format**(ModelFormat): 模型格式，默认为PADDLE格式

### Predict函数

```c#
fastdeploy.ClassifyResult Predict(OpenCvSharp.Mat im)
```

> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
>> * **im**(Mat): 输入图像，注意需为HWC，BGR格式
>>
> **返回值**
>
>> * **result**: 分类结果，包括label_id，以及相应的置信度, ClassifyResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)


## 6. 其它文档
- [FastDeploy部署PaddleClas模型概览](../../)
- [PaddleClas Python部署](../python)
- [PaddleClas C++ 部署](../cpp)
- [PaddleClas C 部署](../c)
