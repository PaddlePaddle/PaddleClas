# CCSSL:Class-Aware Contrastive Semi-Supervised Learning
论文出处：[https://arxiv.org/abs/2203.02261](https://arxiv.org/abs/2203.02261)

## 目录
* 1. 原理介绍
* 2. 精度指标
* 3. 数据准备
* 4. 模型训练
* 5. 模型评估与推理部署
* 5.1 模型评估
* 5.2 模型推理
* * 5.2.1 推理模型准备
* * 5.2.2 基于Python预测引擎推理
* * 5.2.3 基于C++预测引擎推理
* 5.4 服务化部署
* 5.5 端侧部署
* 5.6 Paddle2ONNX模型转换与预测
* 6. 参考文献

## 1. 原理介绍
作者提出了一种新颖的半监督学习方法。对有标签的数据进行数据训练的同时，对无标签数据进行一种弱增强和两种强增强。如果若增强的分类结果大于阈值，则弱数据增强的输出标签作为伪标签。通过伪标签，制作一个仅包含类级信息的监督对比矩阵。然后，通过对分布外数据的图像级对比形成类感知对比矩阵，以减少确认偏差。通过应用重新加权模块，将学习重点放在干净的数据上，并获得最终的目标矩阵。此外，特征亲和矩阵由两个强大的增强视图组成。通过最小化亲和矩阵和目标矩阵之间的交叉熵来制定用于未标记数据的类感知对比模块。模型的流程图如下

## 2. 精度指标
以下表格总结了复现的CCSSL在Cifar10数据集上的精度指标。
<table>
    <tr>
        <td>Labels</td>
        <td>40</td>
        <td>250</td>
        <td>4000</td>
    </tr>
    <tr>
        <td>pytorch版本</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>paddle版本</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>
cifar10上，paddle版本的配置文件及训练好的模型如下表所示
<table>
    <tr>
        <td>label</td>
        <td>配置文件地址</td>
        <td>模型下载链接</td>
    </tr>
    <tr>
        <td>40</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>paddle版本</td>
        <td></td>
        <td></td>
    </tr>
</table>
上表中的配置是基于4卡GPU训练的。
**接下来** 以 `FixMatch_CCSSL_cifar10_40000_4gpu.yaml` 配置和训练好的模型文件为例，展示在cifar10数据集上进行训练，测试，推理的过程。

## 3. 数据准备
cifar10数据在训练过程中会自动下载到默认缓存路径 `~/.cache/paddle/dataset/cifar/`中，请在训练过程中保持网络畅通。

## 4. 模型训练
1. 执行以下命令开始训练

单卡训练执行以下命令
```
python tools/train.py -c ppcls/configs/ssl/FixMatchCCSSL/FixMatchCCSSL_cifar10_4000.yaml
```

4卡训练执行以下操作
```
python -m paddle.distributed.launch --gpus='0,1,2,3' tools/train.py -c ppcls/configs/ssl/FixMatchCCSSL/FixMatchCCSSL_cifar10_4000_4gpu.yaml
```

2. **查看训练日志和保存的模型参数文件** 训练过程中屏幕会实时打印loss等指标信息，同时会保存日志文件 `train.log` ，模型参数文件 `*.pdparams`，优化器参数文件 `*.pdopt` 等内容到`Global.output_dir`指定的文件夹下，默认在 `PaddleClas/output/WideResNet/`文件夹下。

## 5. 模型评估与推理部署
### 5.1 模型评估
准备用于评估的 `*.pdparams` 模型参数文件，可以使用训练好的模型，也可以使用 *4. 模型训练* 中保存的模型。
* 以训练过程中保存的 `best_model_ema.ema.pdparams`为例，执行如下命令即可进行评估。
```
python3.7 tools/eval.py -c ppcls/configs/ssl/FixMatchCCSSL/FixMatchCCSSL_cifar10_4000_4gpu.yaml -o Global.pretrained_model="./output/WideResNet/best_model_ema.ema"
```

* 以训练好的模型为例，下载提供的已经训练好的模型，到 `PaddleClas/pretrained_models` 文件夹中，执行如下命令即可进行评估。
```
# 下载模型
cd PaddleClas
mkdir pretrained_models
cd pretrained_models
wget 
cd ..
# 评估
python3.7 tools/eval.py -c ppcls/configs/ssl/FixMatch_CCSSL_cifar10_4000.yaml -o Global.pretrained_model=""
```
**注：** `pretrained_model` 后填入的地址不需要加 `.pdparams`后缀，在程序运行时会自动补上。

* 查看输出结果
```
[2022/12/08 09:36:13] ppcls INFO: [Eval][Epoch 0][Iter: 0/157]CELoss: 0.00999, loss: 0.00999, top1: 1.00000, top5: 1.00000, batch_cost: 5.11046s, reader_cost: 1.22196, ips: 12.52334 images/sec
[2022/12/08 09:36:13] ppcls INFO: [Eval][Epoch 0][Iter: 20/157]CELoss: 0.04825, loss: 0.04825, top1: 0.95164, top5: 1.00000, batch_cost: 0.02071s, reader_cost: 0.00207, ips: 3089.66447 images/sec
[2022/12/08 09:36:14] ppcls INFO: [Eval][Epoch 0][Iter: 40/157]CELoss: 0.03500, loss: 0.03500, top1: 0.95084, top5: 1.00000, batch_cost: 0.02155s, reader_cost: 0.00108, ips: 2970.07129 images/sec
[2022/12/08 09:36:14] ppcls INFO: [Eval][Epoch 0][Iter: 60/157]CELoss: 0.26421, loss: 0.26421, top1: 0.94928, top5: 0.99949, batch_cost: 0.02048s, reader_cost: 0.00151, ips: 3124.81965 images/sec
[2022/12/08 09:36:14] ppcls INFO: [Eval][Epoch 0][Iter: 80/157]CELoss: 0.16254, loss: 0.16254, top1: 0.95332, top5: 0.99942, batch_cost: 0.02124s, reader_cost: 0.00117, ips: 3013.43961 images/sec
[2022/12/08 09:36:15] ppcls INFO: [Eval][Epoch 0][Iter: 100/157]CELoss: 0.15471, loss: 0.15471, top1: 0.95374, top5: 0.99923, batch_cost: 0.02046s, reader_cost: 0.00098, ips: 3128.15428 images/sec
[2022/12/08 09:36:15] ppcls INFO: [Eval][Epoch 0][Iter: 120/157]CELoss: 0.05237, loss: 0.05237, top1: 0.95493, top5: 0.99935, batch_cost: 0.02061s, reader_cost: 0.00084, ips: 3106.03867 images/sec
[2022/12/08 09:36:16] ppcls INFO: [Eval][Epoch 0][Iter: 140/157]CELoss: 0.03242, loss: 0.03242, top1: 0.95601, top5: 0.99945, batch_cost: 0.02084s, reader_cost: 0.00075, ips: 3071.00311 images/sec
[2022/12/08 09:36:16] ppcls INFO: [Eval][Epoch 0][Avg]CELoss: 0.16041, loss: 0.16041, top1: 0.95610, top5: 0.99950
```
默认评估日志保存在 `PaddleClas/output/WideResNetCCSSL/eval.log`中，可以看到我们提供的模型在cifar10数据集上的评估指标为top1: 95.61, top5: 99.95

### 5.2 模型推理
#### 5.2.1 推理模型准备
将训练过程中保存的模型文件转成inference模型，同样以 `best_model_ema.ema_pdparams`为例，执行以下命令进行转换
```
python3.7 tools/export_model.py \
-c ppcls/configs/ssl/FixMatch_CCSSL/FixMatch_CCSSL_cifar10_4000.yaml \
-o Global.pretrained_model="output/WideResNetCCSSL/best_model_ema.ema" \
-o Global.save_inference_fir="./deploy/inference"
```

#### 5.2.2 基于 Python 预测引擎推理
1. 修改 `PaddleClas/deploy/configs/inference_cls.yaml`
* * 将`infer_imgs:` 后的路径段改为 query 文件夹下的任意一张图片路径（下方配置使用的是`demo.jpg`图片的路径）
* * 将`rec_inference_model.dir:` 后的字段改为解压出来的 inference 模型文件夹路径
* * 将`transform_ops:` 字段下的预处理配置改为 `FixMatch_CCSSL_cifar10_40000.yaml` 中 `Eval.dataset`下的预处理配置
```
Global:
  infer_imgs: "demo"
  rec_inference_model_dir: "./inferece"
  batch_size: 1
  use_gpu: False
  enable_mkldnn: True
  cpu_num_threads: 10
  enable_benchmark: False
  use_fp16: False
  ir_optim: True
  use_tensorrt: False
  gpu_mem: 8000
  enable_profile: False

RecPreProcess:
  transform_ops:
   -  NormalizeImage:
        scale: 1.0/255.0
        mean: [0.4914, 0.4822, 0.4465]
        std: [0.2471, 0.2435, 0.2616]
        order: hwc
PostProcess: null
```

2. 执行推理命令
```
cd ./deploy/
python3.7 python/predict_rec.py -c ./configs/inference_rec.yaml
```

3. 查看输出结果，实际结果为一个长度为10的向量，表示图像分类的结果，如
```

```

#### 5.2.3 基于C++预测引擎推理
PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](https://github.com/zh-hike/PaddleClas/blob/develop/docs/zh_CN/deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考基于 Visual Studio 2019 Community CMake 编译指南完成相应的预测库编译和模型预测工作。

### 5.4 服务化部署
Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考Paddle Serving 代码仓库。

## 5.5 端侧部署
Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考Paddle Lite 代码仓库。

PaddleClas 提供了基于 Paddle Lite 来完成模型[端侧部署](https://github.com/zh-hike/PaddleClas/blob/develop/docs/zh_CN/deployment/image_classification/paddle_lite.md)的示例，您可以参考端侧部署来完成相应的部署工作。

## Paddle2ONNX 模型转换与预测
Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考Paddle2ONNX 代码仓库。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考 [Paddle2ONNX](https://github.com/zh-hike/PaddleClas/blob/develop/docs/zh_CN/deployment/image_classification/paddle2onnx.md) 模型转换与预测来完成相应的部署工作。

## 6. 参考资料
1. [CCSSL](https://arxiv.org/abs/2203.02261)