# 开始使用
---
请事先参考[安装指南](install.md)配置运行环境，并根据[数据说明](./data.md)文档准备ImageNet1k数据，本章节下面所有的实验均以ImageNet1k数据集为例。

## 1. Windows或者CPU上训练与评估

如果在windows系统或者CPU上进行训练与评估，推荐使用`tools/train_multi_platform.py`与`tools/eval_multi_platform.py`脚本。

### 1.1 模型训练

准备好配置文件之后，可以使用下面的方式启动训练。

```
python tools/train_multi_platform.py \
    -c configs/ResNet/ResNet50.yaml \
    -o model_save_dir=./output/ \
    -o use_gpu=True
```

其中，`-c`用于指定配置文件的路径，`-o`用于指定需要修改或者添加的参数，`-o model_save_dir=./output/`表示将配置文件中的`model_save_dir`修改为`./output/`。`-o use_gpu=True`表示使用GPU进行训练。如果希望使用CPU进行训练，则需要将`use_gpu`设置为`False`。

也可以直接修改模型对应的配置文件更新配置。具体配置参数参考[配置文档](config.md)。

* 输出日志示例如下：

    * 如果在训练使用了mixup或者cutmix的数据增广方式，那么日志中只会打印出loss(损失)、lr(学习率)以及该minibatch的训练时间。

    ```
    train step:890  loss:  6.8473 lr: 0.100000 elapse: 0.157s
    ```

    * 如果训练过程中没有使用mixup或者cutmix的数据增广，那么除了loss(损失)、lr(学习率)以及该minibatch的训练时间之外，日志中也会打印出top-1与top-k(默认为5)的信息。

    ```
    epoch:0    train    step:13    loss:7.9561    top1:0.0156    top5:0.1094    lr:0.100000    elapse:0.193s
    ```

训练期间可以通过VisualDL实时观察loss变化，启动命令如下：

```bash
visualdl --logdir ./scalar --host <host_IP> --port <port_num>
```

### 1.2 模型微调

* 根据自己的数据集配置好配置文件之后，可以通过加载预训练模型进行微调，如下所示。

```
python tools/train_multi_platform.py \
    -c configs/ResNet/ResNet50.yaml \
    -o pretrained_model="./pretrained/ResNet50_pretrained"
```

其中`pretrained_model`用于设置加载预训练权重的地址，使用时需要换成自己的预训练模型权重路径，也可以直接在配置文件中修改该路径。

### 1.3 模型恢复训练

* 如果训练任务因为其他原因被终止，也可以加载断点权重继续训练。

```
python tools/train_multi_platform.py \
    -c configs/ResNet/ResNet50.yaml \
    -o checkpoints="./output/ResNet/0/ppcls"
```

其中配置文件不需要做任何修改，只需要在训练时添加`checkpoints`参数即可，表示加载的断点权重路径，使用该参数会同时加载保存的断点权重和学习率、优化器等信息。


### 1.4 模型评估

* 可以通过以下命令完成模型评估。

```bash
python tools/eval_multi_platform.py \
    -c ./configs/eval.yaml \
    -o ARCHITECTURE.name="ResNet50_vd" \
    -o pretrained_model=path_to_pretrained_models
```

可以更改`configs/eval.yaml`中的`ARCHITECTURE.name`字段和`pretrained_model`字段来配置评估模型，也可以通过-o参数更新配置。

**注意：** 加载预训练模型时，需要指定预训练模型的前缀，例如预训练模型参数所在的文件夹为`output/ResNet50_vd/19`，预训练模型参数的名称为`output/ResNet50_vd/19/ppcls.pdparams`，则`pretrained_model`参数需要指定为`output/ResNet50_vd/19/ppcls`，PaddleClas会自动补齐`.pdparams`的后缀。


## 2. 基于Linux+GPU的模型训练与评估

如果机器环境为Linux+GPU，那么推荐使用PaddleClas 提供的模型训练与评估脚本：`tools/train.py`和`tools/eval.py`，可以更快地完成训练与评估任务。

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

可以通过添加-o参数来更新配置：

```bash
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50_vd.yaml \
        -o use_mix=1 \
        --vdl_dir=./scalar/
```

输出日志信息的格式同上。

### 2.2 模型微调

* 根据自己的数据集配置好配置文件之后，可以通过加载预训练模型进行微调，如下所示。

```
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c configs/ResNet/ResNet50.yaml \
        -o pretrained_model="./pretrained/ResNet50_pretrained"
```

其中`pretrained_model`用于设置加载预训练权重的地址，使用时需要换成自己的预训练模型权重路径，也可以直接在配置文件中修改该路径。

* [30分钟玩转PaddleClas教程](./quick_start.md)中包含大量模型微调的示例，可以参考该章节在特定的数据集上进行模型微调。


### 2.3 模型恢复训练

* 如果训练任务，因为其他原因被终止，也可以加载断点权重继续训练。

```
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c configs/ResNet/ResNet50.yaml \
        -o checkpoints="./output/ResNet/0/ppcls"
```

其中配置文件不需要做任何修改，只需要在训练时添加`checkpoints`参数即可，表示加载的断点权重路径，使用该参数会同时加载保存的模型参数权重和学习率、优化器等信息。


### 2.4 模型评估

* 可以通过以下命令完成模型评估。

```bash
python -m paddle.distributed.launch \
    --selected_gpus="0" \
    tools/eval.py \
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
    --model=模型名字 \
    --pretrained_model=预训练模型路径 \
    --output_path=预测模型保存路径

```
之后，通过预测引擎进行推理：
```bash
python tools/infer/predict.py \
    -m model文件路径 \
    -p params文件路径 \
    -i 图片路径 \
    --use_gpu=1 \
    --use_tensorrt=False
```
更多使用方法和推理方式请参考[分类预测框架](../extension/paddle_inference.md)。
