# 开始使用
---
请事先参考[安装指南](install.md)配置运行环境，并根据[数据说明](./data.md)文档准备ImageNet1k数据，本章节下面所有的实验均以ImageNet1k数据集为例。

## 一、设置环境变量

**设置PYTHONPATH环境变量：**

```bash
export PYTHONPATH=path_to_PaddleClas:$PYTHONPATH
```

## 二、模型训练与评估

PaddleClas 提供模型训练与评估脚本：`tools/train.py`和`tools/eval.py`

### 2.1 模型训练

按照如下方式启动模型训练。

```bash
# PaddleClas通过launch方式启动多卡多进程训练
# 通过设置FLAGS_selected_gpus 指定GPU运行卡号

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50_vd.yaml
```

- 输出日志示例如下：

```
epoch:0    train    step:13    loss:7.9561    top1:0.0156    top5:0.1094    lr:0.100000    elapse:0.193
```

可以通过添加-o参数来更新配置：

```bash
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50_vd.yaml \
        -o use_mix=1

```

- 输出日志示例如下：

```
epoch:0    train    step:522    loss:1.6330    lr:0.100000    elapse:0.210
```

也可以直接修改模型对应的配置文件更新配置。具体配置参数参考[配置文档](config.md)。


### 2.2 模型微调

* [30分钟玩转PaddleClas](./quick_start.md)中包含大量模型微调的示例，可以参考该章节在特定的数据集上进行模型微调。

### 2.3 模型评估

```bash
python tools/eval.py \
    -c ./configs/eval.yaml \
    -o ARCHITECTURE.name="ResNet50_vd" \
    -o pretrained_model=path_to_pretrained_models
```
可以更改configs/eval.yaml中的`ARCHITECTURE.name`字段和pretrained_model字段来配置评估模型，也可以通过-o参数更新配置。

## 三、模型推理

PaddlePaddle提供三种方式进行预测推理，接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python tools/export_model.py \
    -model=模型名字 \
    -pretrained_model=预训练模型路径 \
    -output_path=预测模型保存路径

```
之后，通过预测引擎进行推理：
```bash
python tools/infer/predict.py \
    -m model文件路径 \
    -p params文件路径 \
    -i 图片路径 \
    --use_gpu=1 \
    --use_tensorrt=True
```
更多使用方法和推理方式请参考[分类预测框架](../extension/paddle_inference.md)。
