# PaddleClas C++部署示例

本目录下提供`infer.cc`快速完成ResNet50_vd模型在SOPHGO BM1684x板子上加速部署的示例。

## 1. 部署环境准备
在部署前，需自行编译基于SOPHGO硬件的预测库，参考文档[SOPHGO硬件部署环境](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#算能硬件部署环境)

## 2. 生成基本目录文件

该例程由以下几个部分组成
```text
.
├── CMakeLists.txt
├── build  # 编译文件夹
├── image  # 存放图片的文件夹
├── infer.cc
├── preprocess_config.yaml #示例前处理配置文件
└── model  # 存放模型文件的文件夹
```
## 3. 部署实例


### 3.1 编译并拷贝SDK到thirdpartys文件夹

请参考[SOPHGO部署库编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/sophgo.md)仓库编译SDK，编译完成后，将在build目录下生成fastdeploy-x.x.x目录.

### 3.2 拷贝模型文件，以及配置文件至model文件夹
将Paddle模型转换为SOPHGO bmodel模型，转换步骤参考[文档](../README.md)  
将转换后的SOPHGO bmodel模型文件拷贝至model中  
将前处理配置文件也拷贝到model中  
```bash
cp preprocess_config.yaml ./model
```

### 3.3 准备测试图片至image文件夹
```bash
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
cp ILSVRC2012_val_00000010.jpeg ./images
```

### 3.4 编译example

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/sophgo/cpp

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/sophgo/cpp

cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-x.x.x
make
```

### 3.5 运行例程

```bash
./infer_demo model images/ILSVRC2012_val_00000010.jpeg
```

## 4. 其它文档
- [ResNet50_vd python部署](../python)
- [转换ResNet50_vd SOPHGO模型文档](../README.md)
