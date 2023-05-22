# PaddleClas 昆仑芯XPU Python部署示例
本目录下提供`infer.py`快速完成PaddleClas在昆仑芯XPU上部署的示例.

## 1. 部署环境准备
在部署前，需自行编译基于昆仑芯XPU的FastDeploy python wheel包并安装，参考文档，参考文档[昆仑芯XPU部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)

## 2. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleClas模型列表](../README.md)中下载所需模型.

## 3. 运行部署示例
```bash
# 安装FastDpeloy 预测库 python包（详细文档请参考`部署环境准备`）

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/kunlunxin/python

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/kunlunxin/python

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 在昆仑芯XPU AI 处理器上推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --topk 1
```

运行完成后返回结果如下所示
```bash
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```

## 4. 更多指南
- [PaddleClas系列 Python API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/image_classification.html)
- [FastDeploy部署PaddleClas模型概览](../../)
- [PaddleClas C++ 部署](../cpp)
