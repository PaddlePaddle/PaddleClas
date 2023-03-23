
# PaddleClas 量化模型部署-FastDeploy

FastDeploy已支持部署量化模型,并提供一键模型自动化压缩的工具.
用户可以使用一键模型自动化压缩工具,自行对模型量化后部署, 也可以直接下载FastDeploy提供的量化模型进行部署.

## 1. FastDeploy一键模型自动化压缩工具  

FastDeploy 提供了一键模型自动化压缩工具, 能够简单地通过输入一个配置文件, 对模型进行量化.
详细教程请见: [一键模型自动化压缩工具](https://github.com/PaddlePaddle/FastDeploy/tree/develop/tools/common_tools/auto_compression)。**注意**: 推理量化后的分类模型仍然需要FP32模型文件夹下的inference_cls.yaml文件, 自行量化的模型文件夹内不包含此yaml文件, 用户从FP32模型文件夹下复制此yaml文件到量化后的模型文件夹内即可。

## 2. 下载量化完成的PaddleClas模型  

用户也可以直接下载下表中的量化模型进行部署.(点击模型名字即可下载)

| 模型                 | 量化方式 |
| [ResNet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar)            | 离线量化 |
| [MobileNetV1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/mobilenetv1_ssld_ptq.tar)  |  离线量化 |

量化后模型的Benchmark比较，请参考[量化模型 Benchmark](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/quantize.md)

## 3. 部署量化模型

### 3.1 部署代码
FastDeploy 部署量化模型与部署FP32模型完全一致, 用户只需要将输入的模型换为量化后的模型即可.
如果硬件在量化模型部署过程有特殊处理，也会在文档中特别标明.
因此本目录下，不提供代码文件, 量化模型部署参考对应的硬件部署即可, 具体请点击下一小节里的链接.

### 3.2 支持部署量化模型的硬件  

|硬件类型|该硬件是否支持|使用指南|Python|C++|
|:---:|:---:|:---:|:---:|:---:|
|X86 CPU|✅|[链接](cpu-gpu)|✅|✅|
|NVIDIA GPU|✅|[链接](cpu-gpu)|✅|✅|
|飞腾CPU|✅|[链接](cpu-gpu)|✅|✅|
|ARM CPU|✅|[链接](cpu-gpu)|✅|✅|
|Intel GPU(集成显卡)|✅|[链接](cpu-gpu)|✅|✅|  
|Intel GPU(独立显卡)|✅|[链接](cpu-gpu)|✅|✅|  
|昆仑|✅|[链接](kunlunxin)|✅|✅|
|昇腾|✅|[链接](ascend)|✅|✅|
|瑞芯微|✅|[链接](rockchip)|✅|✅|  
|晶晨|✅|[链接](amlogic)|--|✅|  
|算能|✅|[链接](sophgo)|✅|✅|  
