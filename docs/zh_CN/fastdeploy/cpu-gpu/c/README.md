[English](README.md) | 简体中文
# PaddleClas CPU-GPU C部署示例

本目录下提供`infer.c`来调用C API快速完成PaddleClas模型在CPU/GPU上部署的示例。

## 1. 说明  
PaddleClas支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署图像分类模型.

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库.
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

## 3. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleClas模型列表](../README.md)中下载所需模型.

## 4.运行部署示例
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/cpu-gpu/c

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/cpu-gpu/c

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


# 使用CPU在OpenVINO推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 0
# 使用GPU在TensorRT推理
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 1
```

- 注意，以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考文档: [如何在Windows中使用FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md)  


## 5. PaddleClas C API接口简介
下面提供了PaddleClas的C API简介
- 如果用户想要更换部署后端或进行其他定制化操作, 请查看[C Runtime API](https://baidu-paddle.github.io/fastdeploy-api/c/html/runtime__option_8h.html).
- 更多 PaddleClas C API 请查看 [C PaddleClas API](https://github.com/PaddlePaddle/FastDeploy/blob/develop/c_api/fastdeploy_capi/vision/classification/ppcls/model.h)

### 配置

```c
FD_C_RuntimeOptionWrapper* FD_C_CreateRuntimeOptionWrapper()
```

> 创建一个RuntimeOption的配置对象，并且返回操作它的指针。
>
> **返回**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption对象的指针


```c
void FD_C_RuntimeOptionWrapperUseCpu(
     FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper)
```

> 开启CPU推理
>
> **参数**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption对象的指针

```c
void FD_C_RuntimeOptionWrapperUseGpu(
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int gpu_id)
```
> 开启GPU推理
>
> **参数**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption对象的指针
> * **gpu_id**(int): 显卡号


### 模型

```c

FD_C_PaddleClasModelWrapper* FD_C_CreatePaddleClasModelWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* runtime_option,
    const FD_C_ModelFormat model_format)

```

> 创建一个PaddleClas的模型，并且返回操作它的指针。
>
> **参数**
>
> * **model_file**(const char*): 模型文件路径
> * **params_file**(const char*): 参数文件路径
> * **config_file**(const char*): 配置文件路径，即PaddleClas导出的部署yaml文件
> * **runtime_option**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption的指针，表示后端推理配置
> * **model_format**(FD_C_ModelFormat): 模型格式
>
> **返回**
> * **fd_c_ppclas_wrapper**(FD_C_PaddleClasModelWrapper*): 指向PaddleClas模型对象的指针


### 读写图像

```c
FD_C_Mat FD_C_Imread(const char* imgpath)
```

> 读取一个图像，并且返回cv::Mat的指针。
>
> **参数**
>
> * **imgpath**(const char*): 图像文件路径
>
> **返回**
>
> * **imgmat**(FD_C_Mat): 指向图像数据cv::Mat的指针。


```c
FD_C_Bool FD_C_Imwrite(const char* savepath,  FD_C_Mat img);
```

> 将图像写入文件中。
>
> **参数**
>
> * **savepath**(const char*): 保存图像的路径
> * **img**(FD_C_Mat): 指向图像数据的指针
>
> **返回**
>
> * **result**(FD_C_Bool): 表示操作是否成功


### Predict函数

```c
FD_C_Bool FD_C_PaddleClasModelWrapperPredict(
    __fd_take FD_C_PaddleClasModelWrapper* fd_c_ppclas_wrapper, FD_C_Mat img,
    FD_C_ClassifyResult* fd_c_ppclas_result)
```
>
> 模型预测接口，输入图像直接并生成分类结果。
>
> **参数**
> * **fd_c_ppclas_wrapper**(FD_C_PaddleClasModelWrapper*): 指向PaddleClas模型的指针
> * **img**（FD_C_Mat）: 输入图像的指针，指向cv::Mat对象，可以调用FD_C_Imread读取图像获取
> * **fd_c_ppclas_result**（FD_C_ClassifyResult*): 分类结果，包括label_id，以及相应的置信度, ClassifyResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)


### Predict结果

```c
void FD_C_ClassifyResultStr(
    FD_C_ClassifyResult* fd_c_classify_result， char* str_buffer);
```
>
> 打印结果
>
> **参数**
> * **fd_c_classify_result**(FD_C_ClassifyResult*): 指向FD_C_ClassifyResult对象的指针
> * **str_buffer**(char*): 保存结果数据信息的字符串


## 6. 其它文档

- [FastDeploy部署PaddleClas模型概览](../../)
- [PaddleClas Python部署](../python)
- [PaddleClas C++ 部署](../cpp)
- [PaddleClas C# 部署](../csharp)
