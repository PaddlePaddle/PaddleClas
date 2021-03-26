
## 介绍
复杂的模型有利于提高模型的性能，但也导致模型中存在一定冗余，模型量化将全精度缩减到定点数减少这种冗余，达到减少模型计算复杂度，提高模型推理性能的目的。
模型量化可以在基本不损失模型的精度的情况下，将FP32精度的模型参数转换为Int8精度，减小模型参数大小并加速计算，使用量化后的模型在移动端等部署时更具备速度优势。

本教程将介绍如何使用飞桨模型压缩库PaddleSlim做PaddleClas模型的压缩。
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 集成了模型剪枝、量化（包括量化训练和离线量化）、蒸馏和神经网络搜索等多种业界常用且领先的模型压缩功能，如果您感兴趣，可以关注并了解。

在开始本教程之前，建议先了解[PaddleClas模型的训练方法](../../../docs/zh_CN/tutorials/quick_start.md)以及[PaddleSlim](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)


## 快速开始
量化多适用于轻量模型在移动端的部署，当训练出一个模型后，如果希望进一步的压缩模型大小并加速预测，可使用量化的方法压缩模型。

模型量化主要包括五个步骤：
1. 安装 PaddleSlim
2. 准备训练好的模型
3. 量化训练
4. 导出量化推理模型
5. 量化模型预测部署

### 1. 安装PaddleSlim

* 可以通过pip install的方式进行安装。

```bash
pip3.7 install paddleslim==2.0.0
```

* 如果获取PaddleSlim的最新特性，可以从源码安装。

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python3.7 setup.py install
```

### 2. 准备训练好的模型

PaddleClas提供了一系列训练好的[模型](../../../docs/zh_CN/models/models_intro.md)，如果待量化的模型不在列表中，需要按照[常规训练](../../../docs/zh_CN/tutorials/getting_started.md)方法得到训练好的模型。

### 3. 量化训练
量化训练包括离线量化训练和在线量化训练，在线量化训练效果更好，需加载预训练模型，在定义好量化策略后即可对模型进行量化。


量化训练的代码位于`deploy/slim/quant/quant.py` 中，训练指令如下：

* CPU/单机单卡启动

```bash
python3.7 deploy/slim/quant/quant.py \
    -c configs/MobileNetV3/MobileNetV3_large_x1_0.yaml \
    -o pretrained_model="./MobileNetV3_large_x1_0_pretrained"
```

* 单机单卡/单机多卡/多机多卡启动

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    deploy/slim/quant/quant.py \
        -c configs/MobileNetV3/MobileNetV3_large_x1_0.yaml \
        -o pretrained_model="./MobileNetV3_large_x1_0_pretrained"
```


* 下面是量化`MobileNetV3_large_x1_0`模型的训练示例脚本。

```bash
# 下载预训练模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams
# 启动训练，这里如果因为显存限制，batch size无法设置过大，可以将batch size和learning rate同比例缩小。
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    deploy/slim/quant/quant.py \
        -c configs/MobileNetV3/MobileNetV3_large_x1_0.yaml \
        -o pretrained_model="./MobileNetV3_large_x1_0_pretrained"
        -o LEARNING_RATE.params.lr=0.13 \
        -o epochs=100
```

### 4. 导出模型

在得到量化训练保存的模型后，可以将其导出为inference model，用于预测部署：

```bash
python3.7 deploy/slim/quant/export_model.py \
    -m MobileNetV3_large_x1_0 \
    -p output/MobileNetV3_large_x1_0/best_model/ppcls \
    -o ./MobileNetV3_large_x1_0_infer/ \
    --img_size=224 \
    --class_dim=1000
```


### 5. 量化模型部署

上述步骤导出的量化模型，参数精度仍然是FP32，但是参数的数值范围是int8，导出的模型可以通过PaddleLite的opt模型转换工具完成模型转换。
量化模型部署的可参考 [移动端模型部署](../../lite/readme.md)


## 量化训练超参数建议

* 量化训练时，建议加载常规训练得到的预训练模型，加速量化训练收敛。
* 量化训练时，建议初始学习率修改为常规训练的`1/20~1/10`，同时将训练epoch数修改为常规训练的`1/5~1/2`，学习率策略方面，加上Warmup，其他配置信息不建议修改。
