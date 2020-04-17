# 模型微调

本文档将介绍如何使用PaddleClas进行模型微调（finetune）
模型微调使用PaddleClas提供的预训练模型，可以节省从头训练的计算资源和时间，并提高准确率。

> 在使用ResNet50_vd_ssld蒸馏模型对flower102数据进行模型微调，仅需要3分钟（V100 单卡）top1即可达到94.96%


模型微调大致包括如下四个步骤：
- 初始化预训练模型
- 剔除FC层
- 更新参数
- 新的训练策略


## 初始化预训练模型

这里我们以ResNet50_vd和ResNet50_vd_ssld预训练模型对flower102数据集进行微调

ResNet50_vd： 在ImageNet1k数据集上训练 top1 acc：79.1% 模型详细信息参考[模型库](https://paddleclas.readthedocs.io/zh_CN/latest/models/ResNet_and_vd.html)

ResNet50_vd_ssld： 在ImageNet1k数据集训练的蒸馏模型 top1： 82.4% 模型详细信息参考[模型库](https://paddleclas.readthedocs.io/zh_CN/latest/models/ResNet_and_vd.html)

flower数据集相关信息参考[数据文档](data.md)

指定pretrained_model参数初始化预训练模型
ResNet50_vd：

```bash
python -m paddle.distributed.launch \
    --selected_gpus="0" \
    tools/train.py \
        -c ./configs/finetune/ResNet50_vd_finetune.yaml
        -o pretrained_model= ResNet50_vd预训练模型
```

ResNet50_vd_ssld：

```bash
python -m paddle.distributed.launch \
    --selected_gpus="0" \
    tools/train.py \
        -c ./configs/finetune/ResNet50_vd_ssld_finetune.yaml
        -o pretrained_model= ResNet50_vd_ssld预训练模型
```


## 剔除FC层

由于新的数据集类别数（Flower102：102类）和ImgaeNet1k数据（1000类）不一致，一般需要对分类网络的最后FC层进行调整，PaddleClas默认剔除所有形状不一样的层

```python
#excerpt from PaddleClas/ppcls/utils/save_load.py

def load_params(exe, prog, path):
    # ...

    ignore_set = set()
    state = _load_state(path)

    # 剔除预训练模型和模型间形状不一致的参数

    all_var_shape = {}
    for block in prog.blocks:
        for param in block.all_parameters():
            all_var_shape[param.name] = param.shape
    ignore_set.update([
        name for name, shape in all_var_shape.items()
        if name in state and shape != state[name].shape
    ])

    # 用于迁移学习的代码段已被省略 ...

    if len(ignore_set) > 0:
        for k in ignore_set:
            if k in state:
                # 剔除参数
                del state[k]
    fluid.io.set_program_state(prog, state)
```
在将shape不一致的层进行剔除正确加载预训练模型后，我们要选择需要更新的参数来让优化器进行参数更新。

## 更新参数

首先，分类网络中的卷积层大致可以分为

- ```浅层卷积层```：用于提取基础特征
- ```深层卷积层```：用于提取抽象特征
- ```FC层```：进行特征组合

其次，在衡量数据集大小差别和数据集的相似程度后，我们一般遵循如下的规则进行参数更新：

- 1. 新的数据集很小，在类别，具体种类上和原数据很像。由于新数据集很小，这里可能出现过拟合的问题；由于数据很像，可以认为预训练模型的深层特征仍然会起作用，只需要训练一个最终的```FC层```即可。
- 2. 新的数据集很大，在类别，具体种类上和原数据很像。推荐训练网络中全部层的参数。
- 3. 新的数据集很小但是和原数据不相像，可以冻结网络中初始层的参数更新```stop_gradient=True```，对较高层进行重新训练。
- 4. 新的数据集很大但是和原数据不相像，这时候预训练模型可能不会生效，需要从头训练。

PaddleClas模型微调默认更新所有层参数。

## 新的训练策略

1. 学习率
由于已经加载了预训练模型，对于从头训练的随机初始化参数来讲，模型中的参数已经具备了一定的分类能力，所以建议使用与从头训练相比更小的学习率，例如减小10倍。
2. 类别数和总图片数调整为新数据集数据
3. 调整训练轮数，由于不需要从头开始训练，一般相对减少模型微调的训练轮数

## 模型微调结果

在使用ResNet50_vd预训练模型对flower102数据进行模型微调后，top1 acc 达到 92.71%
在使用ResNet50_vd_ssld预训练模型对flower102数据进行模型微调后，top1 acc 达到94.96%
