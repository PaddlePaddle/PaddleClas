# PaddleClas Python部署示例

## 1. 部署环境准备
在部署前，需确认以下两个步骤
- 1. 在部署前，需自行编译基于RKNPU2的Python预测库，参考文档[RKNPU2部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)
- 2. 同时请用户参考[FastDeploy RKNPU2资源导航](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/rknpu2.md)

## 2. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以参考[RKNPU2模型转换](../README.md), 来准备模型.

## 3. 部署示例

本目录下提供`infer.py`快速完成 ResNet50_vd 在RKNPU上部署的示例

```bash
# 安装FastDpeloy RKNPU2 python包（详细文档请参考`部署环境准备`）

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/rockchip/rknpu2/python

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/rockchip/rknpu2/python

# 下载图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 推理
python3 infer.py --model_file ./ResNet50_vd_infer/ResNet50_vd_infer_rk3588.rknn  --config_file ResNet50_vd_infer/inference_cls.yaml  --image ILSVRC2012_val_00000010.jpeg

# 运行完成后返回结果如下所示
ClassifyResult(
label_ids: 153,
scores: 0.684570,
)
```

## 4.注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，需要先调用DisablePermute(C++)或`disable_permute(Python)，在预处理阶段禁用数据格式的转换。

## 5. 其它文档
- [ResNet50_vd C++部署](../cpp)
- [转换ResNet50_vd RKNN模型文档](../README.md)
