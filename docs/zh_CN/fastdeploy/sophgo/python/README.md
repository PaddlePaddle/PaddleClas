# PaddleClas Python部署示例
本目录下提供`infer.py`快速完成 ResNet50_vd 在SOPHGO TPU上部署的示例.


## 1. 部署环境准备

在部署前，需自行编译基于算能硬件的FastDeploy python wheel包并安装，参考文档[算能硬件部署环境](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#算能硬件部署环境)

## 2.运行部署示例
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/sophgo/python

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/sophgo/python

# 下载图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 推理转换好的模型
# 手动设置推理使用的模型、配置文件和图片路径
python3 infer.py --auto False --model_file ./bmodel/resnet50_1684x_f32.bmodel  --config_file ResNet50_vd_infer/inference_cls.yaml  --image ILSVRC2012_val_00000010.jpeg
# 自动完成下载数据-模型编译-推理，不需要设置模型、配置文件和图片路径
python3 infer.py --auto True --model '' --config_file '' --image ''

# 运行完成后返回结果如下所示
ClassifyResult(
label_ids: 153,
scores: 0.684570,
)
```

## 4. 其它文档
- [ResNet50_vd C++部署](../python)
- [转换ResNet50_vd SOPHGO模型文档](../README.md)
