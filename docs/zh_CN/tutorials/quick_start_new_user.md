# 快速上手文档（新手用户）

此教程主要针对初级用户，即深度学习相关理论知识处于入门阶段，具有一定的python基础，能够阅读简单的代码度度用户。此内容主要包括利用PaddleClas进行网络训练及模型预测，具体内容如下：

- [基础知识](#一、基础知识)
- [环境安装与配置](#一、环境安装与配置)
- [数据集的准备与处理](#二、数据的准备与处理)
- [模型训练](#三、模型训练)
- [模型预测](#四、模型预测)

## 一、基础知识

图像分类顾名思义就是一个模式分类问题，是计算机视觉中最基础的任务，它的目标是将不同的图像，划分到不同的类别。以下会对整个模型训练过程中需要了解到的一些概念做简单的解释，希望能够对初次体验PaddleClas的你有所帮助

- train/val/test dataset分别代表模型的训练集、验证集和测试集

  - 训练集（train dataset）：用来训练模型，使模型能够识别不同类型的特征
  - 验证集（val dataset）：练过程中的测试集，方便训练过程中查看模型训练程度
  - 测试集（test dataset）：训练模型结束后，用于评价模型结果的测试集

- 预训练模型

  通过某个较大的数据集训练好的模型，即被预置了参数的权重，这样可以使模型在新的数据集上更快收敛。并尤其是对一些训练数据比较稀缺的任务，在神经网络参数十分庞大的情况下，仅仅依靠任务自身的训练数据可能无法训练充分，加载预训练模型的方法可以认为是让模型基于一个更好的初始状态进行学习，从而能够达到更好的性能。

- 迭代轮数（epoch）

  模型训练迭代的总轮数(模型对训练集全部样本过一遍即为一个epoch。当测试错误率和训练错误率相差较小时，可认为当前迭代次数合适；当测试错误率先变小后变大时则说明迭代次数过大了，需要减小迭代次数，否则容易出现过拟合

- 损失函数（Loss Function）

  训练过程中，衡量模型输出（预测值）与真实值之间的差异

- 准确率（Acc）

  表示预测正确占总数据的比例

  - Top1 Acc：预测结果中概率最大的所在分类正确，则判定为正确
  - Top5 Acc：预测结果中概率排名前5中有分类正确，则判定为正确

## 二、环境安装与配置

具体安装步骤可详看[安装文档]()。

## 三、数据的准备与处理

* 进入PaddleClas目录。

```bash
## linux or mac， $path_to_PaddleClas表示PaddleClas的根目录，用户需要根据自己的真实目录修改
cd $path_to_PaddleClas
```

* 进入`dataset/flowers102`目录，下载并解压flowers102数据集.

```shell
## linux or mac
cd dataset/flowers102
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
tar -xf 102flowers.tgz
```

没有安装`wget`命令或者windows中下载的话，需要将地址拷贝到浏览器中下载，并进行解压到PaddleClas的根目录即可

此时flowers102数据集存放在**dataset/flowers102/jpg** 文件夹中，图像示例如下：

<img src="../../images/quick_start/Examples-Flower-102.png" style="zoom:50%;" />

* 制作train/val/test标签文件

```shell
# 根据flower数据集中的定义，将数据集划分成train、valid、test三个数据集
python generate_flowers102_list.py jpg train > train_list.txt
python generate_flowers102_list.py jpg valid > val_list.txt
python generate_flowers102_list.py jpg test > extra_list.txt
```

**注意**：生成的三个.txt文件中每行格式：**图像相对路径** **图像的label_id**

* 返回`PaddleClas`根目录

```shell
# linux or mac
cd ../../
# windoes直接打开PaddleCls根目录即可
```

## 四、模型训练

### 预训练模型下载

```shell
# 创建文件夹pretrained文件夹并进入
mkdir pretrained && cd pretrained

# 下载预训练模型
# 下载ResNet50_vd模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
# 下载ShuffleNetV2_x0_25模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_25_pretrained.pdparams

# 回到PaddleCls主目录
cd ..
```

Windows操作如上提示，在PaddleClas根目录下创建相应文件夹，并下载好预训练模型后，放到此文件夹中

### 训练模型

#### 使用CPU进行模型训练

由于使用CPU来进行模型训练，计算速度较慢，因此，此处以ShuffleNetV2_x0_25为例。此模型计算量较小，在CPU上计算速度较快。但是也因为模型较小，训练好的模型精度也不会太高。

##### 不使用预训练模型

```shell
#windows在cmd中进入PaddleCls根目录，执行此命令
python tools/train.py -c ./configs/quick_start/ShuffleNetV2_x0_25.yaml
```

- `-c` 参数是指定训练的配置文件路径，训练的具体超参数可查看`yaml`文件
- `yaml`文`use_gpu` 参数设置为`False`，即使用CPU进行训练（若不设置，此参数默认为`TRUE`）
- `yaml`文件中`epochs`参数设置为20，说明对整个数据集进行20个epoch迭代，预计训练20分钟左右（不同CPU，训练时间略有不同），此时训练模型不充分。若提高训练模型精度，请将此参数设大，如**40**，训练时间也会相应延长

##### 使用预训练模型

```shell
python tools/train.py -c ./configs/quick_start/ShuffleNetV2_x0_25.yaml  -o pretrained_model="pretrained/ShuffleNetV2_x0_25_pretrained"
```

- `-o` 参数加入预训练模型地址，注意：预训练模型路径不要加上：`.pdparams`

可以使用将使用与不使用预训练模型训练进行对比，观察loss的下降情况。

#### 使用GPU进行模型训练

由于GPU训练速度更快，可以使用更复杂模型，因此以ResNet50_vd为例。与ShuffleNetV2_x0_25相比，此模型计算量较大， 训练好的模型精度也会更高。

##### 不使用预训练模型

```shell
#for linux or mac，GPU训练暂不支持windows
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c ./configs/quick_start/ResNet50_vd.yaml
```

- `export CUDA_VISIBLE_DEVICES=0` 指定此训练程序可见的GPU为0号卡

训练完成后，验证集的`Top1 Acc`曲线如下所示，最高准确率为0.2735。训练精度曲线下图所示

![image-20210329103510936](../../images/quick_start/r50_vd_acc.png)

##### 使用预训练模型进行训练

基于ImageNet1k分类预训练模型进行微调，训练脚本如下所示

```shell
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c ./configs/quick_start/ResNet50_vd_finetune.yaml
```

**注**：

- 此训练脚本使用GPU，如使用CPU可按照[使用CPU进行模型训练](#使用CPU进行模型训练)所示，进行修改
- 与[不使用预训练模型](#不使用预训练模型)的**yaml**文件的主要不同，此**ymal**文件中加入 **pretrained_model** 参数，此参数指明预训练模型的位置

验证集的`Top1 Acc`曲线如下所示，最高准确率为0.9402，加载预训练模型之后，flowers102数据集精度大幅提升，绝对精度涨幅超过65%。

![image-20210329105947361](../../images/quick_start/r50_vd_pretrained_acc.png)

## 五、模型预测

训练完成后预测代码如下

```shell
cd $path_to_PaddleClas
python tools/infer/infer.py --model ShuffleNetV2_x0_25 -i $image_path_dir --pretrained_model $path_pretrained_model  --use_gpu False
```

其中主要参数如下：

- `-model`：训练时使用擦网络模型，如 ShuffleNetV2_x0_25、ResNet50_vd，具体可查看训练时`yaml`文件中**ARCHITECTURE**下 **name**参数的值
- `-i`：图像文件存放的目录
- `--pretrained_model`： 存放的模型权重位置。上述[CPU训练过程](#使用CPU进行模型训练)中，最优模型存放位置如下：`$path_to_PaddleClas/output/ResNet18_vd/best_model/ppcls.pdparams`，此时此参数应如下填写：`$path_to_PaddleClas/output/ResNet18_vd/best_model/ppcls`，去掉`.pdparams`
- `--use_gpu`：是否使用GPU

运行成功后，示例结果如下：

```txt
Current image file: dataset/flowers102/jpg/image_03946.jpg
        top1, class id: 124, probability: 0.2043
        top2, class id: 281, probability: 0.1033
        top3, class id: 458, probability: 0.0505
        top4, class id: 688, probability: 0.0379
        top5, class id: 789, probability: 0.0357
Current image file: dataset/flowers102/jpg/image_02480.jpg
        top1, class id: 264, probability: 0.0055
        top2, class id: 570, probability: 0.0041
        top3, class id: 795, probability: 0.0037
        top4, class id: 789, probability: 0.0037
        top5, class id: 268, probability: 0.0033
Current image file: dataset/flowers102/jpg/image_00297.jpg
        top1, class id: 264, probability: 0.0035
        top2, class id: 500, probability: 0.0029
        top3, class id: 65, probability: 0.0027
        top4, class id: 2, probability: 0.0024
        top5, class id: 613, probability: 0.0023
```
