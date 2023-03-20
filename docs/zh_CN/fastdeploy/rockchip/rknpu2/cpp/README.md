# PaddleClas RKNPU2 C++部署示例

本目录下提供`infer.cc`, 供用户完成PaddleClas模型在RKNPU2的部署.

## 1. 部署环境准备
在部署前，需确认以下两个步骤
- 1. 在部署前，需自行编译基于RKNPU2的预测库，参考文档[RKNPU2部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)
- 2. 同时请用户参考[FastDeploy RKNPU2资源导航](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/rknpu2.md)

## 2.部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以参考[RKNPU2模型转换](../README.md), 来准备模型.

## 3.部署示例

### 3.1 生成基本目录文件

该例程由以下几个部分组成
```text
.
├── CMakeLists.txt
├── build  # 编译文件夹
├── images  # 存放图片的文件夹
├── infer.cc
├── ppclas_model_dir  # 存放模型文件的文件夹
└── thirdpartys  # 存放sdk的文件夹
```

首先需要先生成目录结构
```bash
mkdir build
mkdir images
mkdir ppclas_model_dir
mkdir thirdpartys
```

### 3.2 编译

#### 3.2.1 编译并拷贝SDK到thirdpartys文件夹

请参考[RK2代NPU部署库编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/rknpu2.md)仓库编译SDK，编译完成后，将在build目录下生成fastdeploy-x.x.x目录，请移动它至thirdpartys目录下.

#### 3.2.2 拷贝模型文件，以及配置文件至model文件夹
在Paddle动态图模型 -> Paddle静态图模型 -> ONNX模型的过程中，将生成ONNX文件以及对应的yaml配置文件，请将配置文件存放到model文件夹内。
转换为RKNN后的模型文件也需要拷贝至model，转换方案: ([ResNet50_vd RKNN模型](../README.md))。

#### 3.2.3 准备测试图片至image文件夹
```bash
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
```

#### 3.2.4 编译example

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/rockchip/rknpu2/cpp

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/rockchip/rknpu2/cpp


cd build
cmake ..
make -j8
make install
```

#### 3.2.5 运行例程

```bash
cd ./build/install
./rknpu_test ./ppclas_model_dir ./images/ILSVRC2012_val_00000010.jpeg
```

#### 3.2.6 运行结果展示
ClassifyResult(
label_ids: 153,
scores: 0.684570,
)

#### 3.2.7 注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，需要先调用DisablePermute(C++)或`disable_permute(Python)，在预处理阶段禁用数据格式的转换。

## 4. 其它文档
- [ResNet50_vd Python 部署](../python)
- [转换ResNet50_vd RKNN模型文档](../README.md)
