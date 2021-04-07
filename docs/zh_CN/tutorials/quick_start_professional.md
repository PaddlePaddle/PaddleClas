# 30分钟玩转PaddleClas（专业版）

此处提供了专业用户在linux操作系统上使用PaddleClas的快速上手教程，主要内容包括基于CIFAR-100数据集和NUS-WIDE-SCENE数据集，快速体验不同模型的单标签训练及多标签训练、加载不同预训练模型、SSLD知识蒸馏方案和数据增广的效果。请事先参考[安装指南](install.md)配置运行环境和克隆PaddleClas代码。


## 一、数据和模型准备

### 1.1 数据准备


* 进入PaddleClas目录。

```
cd path_to_PaddleClas
```

#### 1.1.1 准备CIFAR100

* 创建并进入`dataset/CIFAR100`目录，下载并解压NUS-WIDE-SCENE数据集。

```shell
mkdir dataset/CIFAR100
cd dataset/CIFAR100
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/CIFAR100.tar
tar -xf CIFAR100.tar
```

#### 1.1.2 准备NUS-WIDE-SCENE

* 创建并进入`dataset/NUS-WIDE-SCENE`目录，下载并解压NUS-WIDE-SCENE数据集。

```shell
mkdir dataset/NUS-WIDE-SCENE
cd dataset/NUS-WIDE-SCENE
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/NUS-SCENE-dataset.tar
tar -xf NUS-SCENE-dataset.tar
```

* 返回`PaddleClas`根目录

```
cd ../../
```

### 1.2 模型准备

通过下面的命令下载所需要的预训练模型。

```bash
mkdir pretrained
cd pretrained
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams
cd ../
```


## 二、模型训练

### 2.1 单标签训练

#### 2.1.1 零基础训练：不加载预训练模型的训练

* 基于ResNet50_vd模型，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml
```


验证集的最高准确率为0.415左右。


#### 2.1.2 迁移学习

* 基于ImageNet1k分类预训练模型ResNet50_vd_pretrained(准确率79.12\%)进行微调，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/professional/ResNet50_vd_CIFAR100_finetune.yaml
```

验证集最高准确率为0.718左右，加载预训练模型之后，CIFAR100数据集精度大幅提升，绝对精度涨幅30\%。

* 基于ImageNet1k分类预训练模型ResNet50_vd_ssld_pretrained(准确率82.39\%)进行微调，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/professional/ResNet50_vd_ssld_CIFAR100_finetune.yaml
```

最终CIFAR100验证集上精度指标为0.73，相对于79.12\%预训练模型的微调结构，新数据集指标可以再次提升1.2\%。

* 替换backbone为MobileNetV3_large_x1_0进行微调，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/professional/MobileNetV3_large_x1_0_CIFAR100_finetune.yaml
```

验证集最高准确率为0.601左右, 较ResNet50_vd低近12%。


### 2.2 多标签训练

* 基于ImageNet1k分类预训练模型进行微调NUS-WIDE-SCENE数据集，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/ResNet50_vd_multilabel.yaml
```

训练10epoch之后，验证集最好的准确率应该在0.72左右。

* 零基础训练(不加载预训练模型)只需要将配置文件中的`pretrained_model`置为`""`即可。


## 三、数据增广

PaddleClas包含了很多数据增广的方法，如Mixup、Cutout、RandomErasing等，具体的方法可以参考[数据增广的章节](../advanced_tutorials/image_augmentation/ImageAugment.md)。

### 数据增广的尝试-Mixup

基于`3.3节`中的训练方法，结合Mixup的数据增广方式进行训练，具体的训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/professional/ResNet50_vd_mixup_CIFAR100_finetune.yaml
```

最终CIFAR100验证集上的精度为0.73，使用数据增广可以使得模型精度再次提升约1.2\%。



* **注意**

    * 其他数据增广的配置文件可以参考`configs/DataAugment`中的配置文件。
    
    * 训练CIFAR100的迭代轮数较少，因此进行训练时，验证集的精度指标可能会有1\%左右的波动。


## 四、知识蒸馏


PaddleClas包含了自研的SSLD知识蒸馏方案，具体的内容可以参考[知识蒸馏章节](../advanced_tutorials/distillation/distillation.md)本小节将尝试使用知识蒸馏技术对MobileNetV3_large_x1_0模型进行训练，使用`2.1.2小节`训练得到的ResNet50_vd模型作为蒸馏所用的教师模型，首先将`2.1.2小节`训练得到的ResNet50_vd模型保存到指定目录，脚本如下。

```shell
cp -r output/ResNet50_vd/best_model/  ./pretrained/CIFAR100_R50_vd_final/
```

配置文件中数据数量、模型结构、预训练地址以及训练的数据配置如下：

```yaml
total_images: 50000
ARCHITECTURE:
    name: 'ResNet50_vd_distill_MobileNetV3_large_x1_0'
pretrained_model:
    - "./pretrained/CIFAR100_R50_vd_final/ppcls"
    - "./pretrained/MobileNetV3_large_x1_0_pretrained/”
```

最终的训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/professional/R50_vd_distill_MV3_large_x1_0_CIFAR100.yaml
```

最终CIFAR100验证集上的精度为64.4\%，使用教师模型进行知识蒸馏，MobileNetV3的精度涨幅4.3\%。

* **注意**

    * 蒸馏过程中，教师模型使用的预训练模型为CIFAR100数据集上的训练结果，学生模型使用的是ImageNet1k数据集上精度为75.32\%的MobileNetV3_large_x1_0预训练模型。
    
    * 该蒸馏过程无须使用真实标签，所以可以使用更多的无标签数据，在使用过程中，可以将无标签数据生成假的train_list.txt，然后与真实的train_list.txt进行合并, 用户可以根据自己的数据自行体验。


## 五、多标签分类

本小结提供了多标签分类的样例，数据集NUS-WIDE-SCENE是数据集NUS-WIDE的一个子集，类别数目为33类，图片总数是17463张。训练脚本如下。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/ResNet50_vd_multilabel.yaml
```

训练10epoch之后，验证集最好的正确率应该在0.72左右。

## 六、模型评估与推理

### 6.1 单标签分类模型评估与推理

#### 6.1.1 单标签分类模型评估。

训练好模型之后，可以通过以下命令实现对模型精度的评估。

```bash
python3 tools/eval.py \
    -c ./configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o pretrained_model="./output/ResNet50_vd/best_model/ppcls"
```

#### 6.1.2 单标签分类模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer/infer.py \
    -i "./dataset/CIFAR100/test/0/0001.png" \
    --model ResNet50_vd \
    --pretrained_model "./output/ResNet50_vd/best_model/ppcls" \
    --use_gpu True
```


#### 6.1.3 单标签分类使用inference模型进行模型推理

通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    --model ResNet50_vd \
    --pretrained_model ./output/ResNet50_vd/best_model/ppcls \
    --output_path ./inference \
    --class_dim 100 \
    --img_size 32
```

其中，参数`--model`用于指定模型名称，`--pretrained_model`用于指定模型文件路径，`--output_path`用于指定转换后模型的存储路径。

* **注意**：
    * `--output_path`表示输出的inference模型文件夹路径，若`--output_path=./inference`，则会在`inference`文件夹下生成`inference.pdiparams`、`inference.pdmodel`和`inference.pdiparams.info`文件。

    * 可以通过设置参数`--img_size`指定模型输入图像的`shape`，默认为`224`，表示图像尺寸为`224*224`，请根据实际情况修改。

上述命令将生成模型结构文件（`inference.pdmodel`）和模型权重文件（`inference.pdiparams`），然后可以使用预测引擎进行推理：

```bash
python3 tools/infer/predict.py \
    --image_file "./dataset/CIFAR100/test/0/0001.png" \
    --model_file "./inference/inference.pdmodel" \
    --params_file "./inference/inference.pdiparams" \
    --use_gpu=True \
    --use_tensorrt=False
```

### 6.2 多标签分类模型评估与预测

#### 6.2.1 多标签分类模型评估

训练好模型之后，可以通过以下命令实现对模型精度的评估。

```bash
python3 tools/eval.py \
    -c ./configs/quick_start/ResNet50_vd_multilabel.yaml \
    -o pretrained_model="./output/ResNet50_vd/best_model/ppcls" 
```

评估指标采用mAP，验证集的mAP应该在0.57左右。

#### 6.2.2 多标签分类模型预测

```bash
python3 tools/infer/infer.py \
    -i "./dataset/NUS-WIDE-SCENE/NUS-SCENE-dataset/images/0199_434752251.jpg" \
    --model ResNet50_vd \
    --pretrained_model "./output/ResNet50_vd/best_model/ppcls" \
    --use_gpu True \
    --multilabel True \
    --class_num 33
```
