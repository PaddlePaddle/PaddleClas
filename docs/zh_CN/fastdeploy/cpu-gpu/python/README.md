# PaddleClas CPU-GPU Python部署示例
本目录下提供`infer.py`快速完成PaddleClas在CPU/GPU上部署的示例.

## 1. 说明  
PaddleClas支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署图像分类模型

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库.

## 3. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleClas模型列表](../README.md)中下载所需模型.

## 4. 运行部署示例
```bash
# 安装FastDpeloy python包（详细文档请参考`部署环境准备`）
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2

# 下载部署示例代码
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/cpu-gpu/python

# 如果您希望从PaddleClas下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleClas.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
git checkout develop
cd PaddleClas/deploy/fastdeploy/cpu-gpu/python

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 在CPU上使用Paddle Inference推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device cpu --backend paddle --topk 1
# 在CPU上使用OenVINO推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device cpu --backend openvino --topk 1
# 在CPU上使用ONNX Runtime推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device cpu --backend ort --topk 1
# 在CPU上使用Paddle Lite推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device cpu --backend pplite --topk 1
# 在GPU上使用Paddle Inference推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --backend paddle --topk 1
# 在GPU上使用Paddle TensorRT推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --backend pptrt --topk 1
# 在GPU上使用ONNX Runtime推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --backend ort --topk 1
# 在GPU上使用Nvidia TensorRT推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --backend trt --topk 1
```

运行完成后返回结果如下所示
```bash
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```

## 5. 部署示例选项说明  

|参数|含义|默认值
|---|---|---|  
|--model|指定模型文件夹所在的路径|None|
|--image|指定测试图片所在的路径|None|  
|--device|指定即将运行的硬件类型，支持的值为`[cpu, gpu]`，当设置为cpu时，可运行在x86 cpu/arm cpu等cpu上|cpu|
|--device_id|使用gpu时, 指定设备号|0|
|--backend|部署模型时使用的后端, 支持的值为`[paddle,pptrt,pplite,ort,openvino,trt]` |openvino|
|--topk|返回的前topk准确率, 支持的为`1,5` |1|

关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

## 6. 更多指南
- [PaddleClas系列 Python API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/image_classification.html)
- [PaddleClas C++ 部署](../cpp)
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
