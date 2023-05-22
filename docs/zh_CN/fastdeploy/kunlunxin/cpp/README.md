# PaddleClas 昆仑芯XPU C++部署示例

本目录下提供`infer.cc`, 供用户完成PaddleClas模型在昆仑芯XPU上的部署.

## 1. 部署环境准备
在部署前，需自行编译基于昆仑芯XPUXPU的预测库，参考文档[昆仑芯XPU部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)

## 2. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleClas模型列表](../README.md)中下载所需模型.

## 3. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.0以上(x.x.x>=1.0.0)

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/kunlunxin/cpp

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/kunlunxin/cpp

mkdir build
cd build
# 使用编译完成的FastDeploy库编译infer_demo
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-kunlunxin
make -j

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 使用昆仑芯XPU部署
./infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg

```
运行完成后返回结果如下所示
```bash
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```

## 4. 更多指南
- [PaddleClas系列 C++ API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1classification.html)
- [FastDeploy部署PaddleClas模型概览](../../)
- [PaddleClas Python部署](../python)
