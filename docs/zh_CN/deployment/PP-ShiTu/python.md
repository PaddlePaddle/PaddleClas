# Python 预测推理

首先请参考文档[环境准备](../../installation.md)配置运行环境。

## 目录

- [1. PP-ShiTu模型推理](#1)
    - [1.1 主体检测模型推理](#1.1)
    - [1.2 特征提取模型推理](#1.2)
    - [1.3 PP-ShiTu PipeLine推理](#1.3)

<a name="1"></a>

## 1. PP-ShiTu模型推理

PP-ShiTu整个Pipeline包含三部分：主体检测、特征提取模型、特征检索。其中主体检测模型、特征提取模型可以单独推理使用。单独使用主体检测详见[主体检测模型推理](#2.1)，特征提取模型单独推理详见[特征提取模型推理](#2.2)， PP-ShiTu整体推理详见[PP-ShiTu PipeLine推理](#2.3)。

<a name="2.1"></a>

### 1.1 主体检测模型推理

进入 PaddleClas 的 `deploy` 目录下：

```shell
cd PaddleClas/deploy
```

准备 PaddleClas 提供的主体检测 inference 模型：

```shell
mkdir -p models
# 下载通用检测 inference 模型并解压
wget -nc -P ./models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
tar -xf ./models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar -C ./models/
```

使用以下命令进行预测：

```shell
python3.7 python/predict_det.py -c configs/inference_det.yaml
```

在配置文件 `configs/inference_det.yaml` 中有以下字段用于配置预测参数：
* `Global.infer_imgs`：待预测的图片文件路径；
* `Global.use_gpu`： 是否使用 GPU 预测，默认为 `True`。

<a name="2.2"></a>

### 1.2 特征提取模型推理

下面以商品图片的特征提取为例，介绍特征提取模型推理。首先进入 PaddleClas 的 `deploy` 目录下：

```shell
cd PaddleClas/deploy
```

准备 PaddleClas 提供的商品特征提取 inference 模型：

```shell
mkdir -p models
# 下载商品特征提取 inference 模型并解压
wget -nc -P ./models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0_infer.tar
tar -xf ./models/general_PPLCNetV2_base_pretrained_v1.0_infer.tar -C ./models/
```

使用以下命令进行预测：

```shell
python3.7 python/predict_rec.py -c configs/inference_rec.yaml
```

上述预测命令可以得到一个 512 维的特征向量，直接输出在在命令行中。

在配置文件 `configs/inference_rec.yaml` 中有以下字段用于配置预测参数：
* `Global.infer_imgs`：待预测的图片文件路径；
* `Global.use_gpu`： 是否使用 GPU 预测，默认为 `True`。

<a name="1.3"></a>

### 1.3 PP-ShiTu PipeLine推理

主体检测、特征提取和向量检索的串联预测，可以参考[图像识别快速开始](../../quick_start/quick_start_recognition.md)。
