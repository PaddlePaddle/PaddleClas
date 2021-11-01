# 30分钟玩转PaddleClas（尝鲜版）

此教程主要针对初级用户，即深度学习相关理论知识处于入门阶段，具有一定的 Python 基础，能够阅读简单代码的用户。此内容主要包括使用 PaddleClas 进行图像分类网络训练及模型预测。

---

## 1. 基础知识

图像分类顾名思义就是一个模式分类问题，是计算机视觉中最基础的任务，它的目标是将不同的图像，划分到不同的类别。以下会对整个模型训练过程中需要了解到的一些概念做简单的解释，希望能够对初次体验 PaddleClas 的你有所帮助：

- train/val/test dataset 分别代表模型的训练集、验证集和测试集：
  - 训练集（train dataset）：用来训练模型，使模型能够识别不同类型的特征；
  - 验证集（val dataset）：训练过程中的测试集，方便训练过程中查看模型训练程度；
  - 测试集（test dataset）：训练模型结束后，用于评价模型结果的测试集。

- 预训练模型

  使用在某个较大的数据集训练好的预训练模型，即被预置了参数的权重，可以帮助模型在新的数据集上更快收敛。尤其是对一些训练数据比较稀缺的任务，在神经网络参数十分庞大的情况下，仅仅依靠任务自身的训练数据可能无法训练充分，加载预训练模型的方法可以认为是让模型基于一个更好的初始状态进行学习，从而能够达到更好的性能。

- 迭代轮数（epoch）

  模型训练迭代的总轮数，模型对训练集全部样本过一遍即为一个 epoch。当测试错误率和训练错误率相差较小时，可认为当前迭代轮数合适；当测试错误率先变小后变大时，则说明迭代轮数过大，需要减小迭代轮数，否则容易出现过拟合。

- 损失函数（Loss Function）

  训练过程中，衡量模型输出（预测值）与真实值之间的差异

- 准确率（Acc）：表示预测正确的样本数占总数据的比例

  - Top1 Acc：预测结果中概率最大的所在分类正确，则判定为正确；
  - Top5 Acc：预测结果中概率排名前 5 中有分类正确，则判定为正确；

## 2. 环境安装与配置

具体安装步骤可详看[Paddle 安装文档](../installation/install_paddle.md)，[PaddleClas 安装文档](../installation/install_paddleclas.md)。

## 3. 数据的准备与处理

进入PaddleClas目录：

```shell
# linux or mac， $path_to_PaddleClas表示PaddleClas的根目录，用户需要根据自己的真实目录修改
cd $path_to_PaddleClas
```

进入 `dataset/flowers102` 目录，下载并解压 flowers102 数据集：

```shell
# linux or mac
cd dataset/flowers102
# 如果希望从浏览器中直接下载，可以复制该链接并访问，然后下载解压即可
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/flowers102.zip
# 解压
unzip flowers102.zip
```

没有安装 `wget` 命令或者在 Windows 中下载的话，需要将地址拷贝到浏览器中下载，并进行解压到 PaddleClas 的根目录即可。

解压完成后，在文件夹下已经生成用于训练和测试的三个 `.txt` 文件：`train_list.txt`（训练集，1020张图）、`val_list.txt`（验证集，1020张图）、`train_extra_list.txt`（更大的训练集，7169张图）。文件中每行格式：**图像相对路径**  **图像的label_id**（注意：中间有空格）。

此时flowers102数据集存放在**dataset/flowers102/jpg** 文件夹中，图像示例如下：

<div align="center">
<img src="../../images/quick_start/Examples-Flower-102.png" width = "800" />
</div>

返回 `PaddleClas` 根目录：

```shell
# linux or mac
cd ../../
# windoes直接打开PaddleClas根目录即可
```

## 4. 模型训练

<a name="4.1"></a>

### 4.1 使用CPU进行模型训练

由于使用CPU来进行模型训练，计算速度较慢，因此，此处以 ShuffleNetV2_x0_25 为例。此模型计算量较小，在 CPU 上计算速度较快。但是也因为模型较小，训练好的模型精度也不会太高。

#### 4.1.1 不使用预训练模型

```shell
# windows在cmd中进入PaddleClas根目录，执行此命令
python tools/train.py -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml
```

- `-c` 参数是指定训练的配置文件路径，训练的具体超参数可查看`yaml`文件
- `yaml`文`Global.device` 参数设置为`cpu`，即使用CPU进行训练（若不设置，此参数默认为`True`）
- `yaml`文件中`epochs`参数设置为20，说明对整个数据集进行20个epoch迭代，预计训练20分钟左右（不同CPU，训练时间略有不同），此时训练模型不充分。若提高训练模型精度，请将此参数设大，如**40**，训练时间也会相应延长

#### 4.1.2 使用预训练模型

```shell
python tools/train.py -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml  -o Arch.pretrained=True
```

- `-o` 参数可以选择为 `True` 或 `False`，也可以是预训练模型存放路径，当选择为 `True` 时，预训练权重会自动下载到本地。注意：若为预训练模型路径，则不要加上：`.pdparams`

可以使用将使用与不使用预训练模型训练进行对比，观察 loss 的下降情况。

### 4.2 使用GPU进行模型训练

由于 GPU 训练速度更快，可以使用更复杂模型，因此以 ResNet50_vd 为例。与 ShuffleNetV2_x0_25 相比，此模型计算量较大，训练好的模型精度也会更高。

首先要设置环境变量，使用 0 号 GPU 进行训练：

- 对于 Linux 用户

  ```shell
  export CUDA_VISIBLE_DEVICES=0
  ```

- 对于 Windows 用户

  ```shell
  set CUDA_VISIBLE_DEVICES=0
  ```

#### 不使用预训练模型

```shell
python3 tools/train.py -c ./ppcls/configs/quick_start/ResNet50_vd.yaml
```

训练完成后，验证集的`Top1 Acc`曲线如下所示，最高准确率为0.2735。训练精度曲线下图所示

<div align="center">
<img src="../../images/quick_start/r50_vd_acc.png"  width = "800" />
</div>

#### 4.2.1 使用预训练模型进行训练

基于 ImageNet1k 分类预训练模型进行微调，训练脚本如下所示

```shell
python3 tools/train.py -c ./ppcls/configs/quick_start/ResNet50_vd.yaml -o Arch.pretrained=True
```

**注**：此训练脚本使用 GPU，如使用 CPU 可按照上文中[4.1 使用CPU进行模型训练](#4.1)所示，进行修改。

验证集的 `Top1 Acc` 曲线如下所示，最高准确率为 `0.9402`，加载预训练模型之后，flowers102 数据集精度大幅提升，绝对精度涨幅超过 65%。

<div align="center">
<img src="../../images/quick_start/r50_vd_pretrained_acc.png"  width = "800" />
</div>

## 5. 模型预测

训练完成后预测代码如下：

```shell
cd $path_to_PaddleClas
python3 tools/infer.py -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml -o Infer.infer_imgs=dataset/flowers102/jpg/image_00001.jpg -o Global.pretrained_model=output/ShuffleNetV2_x0_25/best_model
```

`-i` 输入为单张图像路径，运行成功后，示例结果如下：

`[{'class_ids': [76, 65, 34, 9, 69], 'scores': [0.91762, 0.01801, 0.00833, 0.0071, 0.00669], 'file_name': 'dataset/flowers102/jpg/image_00001.jpg', 'label_names': []}]`

`-i` 输入为图像集所在目录，运行成功后，示例结果如下：

```txt
[{'class_ids': [76, 65, 34, 9, 69], 'scores': [0.91762, 0.01801, 0.00833, 0.0071, 0.00669], 'file_name': 'dataset/flowers102/jpg/image_00001.jpg', 'label_names': []}, {'class_ids': [76, 69, 34, 28, 9], 'scores': [0.77122, 0.06295, 0.02537, 0.02531, 0.0251], 'file_name': 'dataset/flowers102/jpg/image_00002.jpg', 'label_names': []}, {'class_ids': [99, 76, 81, 85, 16], 'scores': [0.26374, 0.20423, 0.07818, 0.06042, 0.05499], 'file_name': 'dataset/flowers102/jpg/image_00003.jpg', 'label_names': []}, {'class_ids': [9, 37, 34, 24, 76], 'scores': [0.17784, 0.16651, 0.14539, 0.12096, 0.04816], 'file_name': 'dataset/flowers102/jpg/image_00004.jpg', 'label_names': []}, {'class_ids': [76, 66, 91, 16, 13], 'scores': [0.95494, 0.00688, 0.00596, 0.00352, 0.00308], 'file_name': 'dataset/flowers102/jpg/image_00005.jpg', 'label_names': []}, {'class_ids': [76, 66, 34, 8, 43], 'scores': [0.44425, 0.07487, 0.05609, 0.05609, 0.03667], 'file_name': 'dataset/flowers102/jpg/image_00006.jpg', 'label_names': []}, {'class_ids': [86, 93, 81, 22, 21], 'scores': [0.44714, 0.13582, 0.07997, 0.0514, 0.03497], 'file_name': 'dataset/flowers102/jpg/image_00007.jpg', 'label_names': []}, {'class_ids': [13, 76, 81, 18, 97], 'scores': [0.26771, 0.1734, 0.06576, 0.0451, 0.03986], 'file_name': 'dataset/flowers102/jpg/image_00008.jpg', 'label_names': []}, {'class_ids': [34, 76, 8, 5, 9], 'scores': [0.67224, 0.31896, 0.00241, 0.00227, 0.00102], 'file_name': 'dataset/flowers102/jpg/image_00009.jpg', 'label_names': []}, {'class_ids': [76, 34, 69, 65, 66], 'scores': [0.95185, 0.01101, 0.00875, 0.00452, 0.00406], 'file_name': 'dataset/flowers102/jpg/image_00010.jpg', 'label_names': []}]
```
其中，列表的长度为 batch_size 的大小。
