# 图像分类模型自动压缩示例

目录：
- [1. 简介](#1-简介)
- [2. Benchmark](#2-benchmark)
- [3. 自动压缩流程](#3-自动压缩流程)
  - [3.1 准备环境](#31-准备准备)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4. 配置文件介绍](#4-配置文件介绍)
- [5. 预测部署](#5-预测部署)
  - [5.1 Python预测推理](#51-python预测推理)
  - [5.2 PaddleLite端侧部署](#52-paddlelite端侧部署)
- [6. FAQ](#6-faq)


## 1. 简介
本示例将以图像分类模型MobileNetV3为例，介绍如何使用PaddleClas中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为量化训练和蒸馏。

## 2. Benchmark

### PaddleClas模型

| 模型 | 策略 | Top-1 Acc | GPU 耗时(ms) | ARM CPU 耗时(ms) | 配置文件 | Inference模型 |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| MobileNetV3_large_x1_0 | Baseline | 75.32 | - | 16.62 | - | [Model](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar) |
| MobileNetV3_large_x1_0 | 量化+蒸馏 | 74.40 | - | 9.85 | [Config](./mbv3_qat_dis.yaml) | [Model](https://paddle-slim-models.bj.bcebos.com/act/MobileNetV3_large_x1_0_QAT.tar) |


- ARM CPU 测试环境：`SDM865(4xA77+4xA55)`


## 3. 自动压缩流程

#### 3.1 准备环境

- python >= 3.6
- PaddlePaddle >= 2.3 （可从[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3

安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

安装paddleslim：
```shell
pip install paddleslim
```

#### 3.2 准备数据集
本案例默认以ImageNet1k数据进行自动压缩实验，如数据集为非ImageNet1k格式数据， 请参考[PaddleClas数据准备文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/data_preparation/classification_dataset.md)。将下载好的数据集放在当前目录下`./ILSVRC2012`。


#### 3.3 准备预测模型
预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`，带`pdmodel`后缀的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

可在[PaddleClas预训练模型库](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)中直接获取Inference模型，具体可参考下方获取MobileNetV3模型示例：

```shell
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
tar -xf MobileNetV3_large_x1_0_infer.tar
```
也可根据[PaddleClas文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/inference_deployment/export_model.md)导出Inference模型。

#### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口 ```paddleslim.auto_compression.AutoCompression``` 对模型进行量化训练和蒸馏。配置config文件中模型路径、数据集路径、蒸馏、量化和训练等部分的参数，配置完成后便可开始自动压缩。

**单卡启动**

```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --save_dir='./save_quant_mobilev3/' --config_path='./configs/mbv3_qat_dis.yaml'
```

**多卡启动**

图像分类训练任务中往往包含大量训练数据，以ImageNet-1k为例，如果使用单卡训练，会非常耗时，使用分布式训练可以达到几乎线性的加速比。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch run.py --save_dir='./save_quant_mobilev3/' --config_path='./configs/mbv3_qat_dis.yaml'
```
多卡训练指的是将训练任务按照一定方法拆分到多个训练节点完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。服务节点在收到所有训练节点传来的梯度后，会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。多卡训练一轮训练能训练```batch size * num gpus```的数据，比如单卡的```batch size```为128，单轮训练的数据量即128，而四卡训练的```batch size```为128，单轮训练的数据量为512。

注意 ```learning rate``` 与 ```batch size``` 呈线性关系，这里单卡 ```batch size``` 为128，对应的 ```learning rate``` 为0.001，那么如果 ```batch size``` 减小4倍改为32，```learning rate``` 也需除以4；多卡时 ```batch size``` 为128，```learning rate``` 需乘上卡数。所以改变 ```batch size``` 或改变训练卡数都需要对应修改 ```learning rate```。

加载训练好的模型进行量化训练时，一般`learning rate`可比原始训练的`learning rate`小10倍。


## 4. 配置文件介绍
自动压缩相关配置主要有：
- 压缩策略配置，如量化（Quantization），知识蒸馏（Distillation），结构化稀疏（ChannelPrune），ASP半结构化稀疏（ASPPrune ），非结构化稀疏（UnstructurePrune）。
- 训练超参配置（TrainConfig）：主要设置学习率、训练次数（epochs）和优化器等。
- 全局配置（Global）：需提供inference模型文件路径，输入名称等信息。

详细介绍可参考[ACT超参详细教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/hyperparameter_tutorial.md)

注意```DataLoader```的使用与```PaddleClas```中的相同，保持与```PaddleClas```中相同配置即可。不同模型```DataLoader```的配置可参考[PaddleClas配置文件](https://github.com/PaddlePaddle/PaddleClas/tree/develop/ppcls/configs/ImageNet)。




## 5. 预测部署
#### 5.1 Python预测推理
Python预测推理可参考：
- [Python部署](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/inference_deployment/python_deploy.md)



#### 5.2 PaddleLite端侧部署
PaddleLite端侧部署可参考：
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/inference_deployment/paddle_lite_deploy.md)

## 6. FAQ
