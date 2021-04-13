# PaddleClas代码解析与社区贡献指南

## 1. 整体代码结构解析

### 1.1 前言

有用户对PaddleClas的代码做了非常详细的解读，可以参考下面的三篇文档。本部分内容大部分也是来自该系列文档，在此感谢[FutureSI](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/76563)的贡献与解读。


* [PaddleClas 分类套件源码解析（一）](https://aistudio.baidu.com/aistudio/projectdetail/1308952)
* [PaddleClas 分类套件源码解析（二）](https://aistudio.baidu.com/aistudio/projectdetail/1315501)
* [PaddleClas 分类套件源码解析（三）](https://aistudio.baidu.com/aistudio/projectdetail/1339544)

### 1.2 代码解析

#### 1.2.1 整体代码和目录概览

* PaddleClas主要代码和目录结构如下

<div align="center">
<img src="../../images/quick_start/community/code_framework.png"  width = "800" />
</div>

* configs 文件夹下存放训练脚本和验证脚本的yaml配置文件，文件按模型类别存放。
* dataset 文件夹下存放数据集和用于处理数据集的脚本。脚本负责将数据集处理为适合Dataloader处理的格式。
* docs 文件夹下存放中英文文档。
* deploy 文件夹存放的是部署工具，支持 Cpp inference、Hub Serveing、Paddle Lite、Slim量化等多种部署方式。
* ppcls 文件夹下存放PaddleClas框架主体。模型结构脚本、数据增强脚本、优化脚本等DL程序的标准流程代码都在这里。
* tools 文件夹下存放用于模型下载、训练、预测的脚本。
* requirements.txt 文件用于安装 PaddleClas 的依赖项。使用pip进行升级安装使用。

#### 1.2.2 训练模块定义

深度学习模型训练流程框图如下。

<div align="center">
<img src="../../images/quick_start/community/train_framework.png"  width = "800" />
</div>

具体地，深度学习模型训练过程中，主要包含以下几个核心模块。

* 数据：对于有监督任务来说，训练数据一般包含原始数据及其标注。在基于单标签的图像分类任务中，原始数据指的是图像数据，而标注则是该图像数据所属的类比。PaddleClas中，训练时需要提供标签文件，形式如下，每一行包含一条训练样本，分别表示图片路径和类别标签，用分隔符隔开（默认为空格）。

```
train/n01440764/n01440764_10026.JPEG 0
train/n01440764/n01440764_10027.JPEG 0
```

在代码`ppcls/data/reader.py`中，包含`CommonDataset`类，继承自`paddle.io.Dataset`，该数据集类可以通过一个键值进行索引并获取指定样本。

对于读入的数据，需要通过数据转换，将原始的图像数据进行转换。训练时，标准的数据预处理包含：`DecodeImage`, `RandCropImage`, `RandFlipImage`, `NormalizeImage`, `ToCHWImage`。在配置文件中体现如下，数据预处理主要包含在`transforms`字段中，以列表形式呈现，会按照顺序对数据依次做这些转换。

```yaml
TRAIN:
    batch_size: 256 # 所有训练设备上的总batch size
    num_workers: 4 # 训练时每块设备上的进程数
    file_list: "./dataset/ILSVRC2012/train_list.txt" # 训练标签文件
    data_dir: "./dataset/ILSVRC2012/" # 训练图片文件夹
    shuffle_seed: 0 # 随机打散的种子数
    transforms:
        - DecodeImage: # 对图像文件进行解码，转成numpy矩阵
            to_rgb: True
            channel_first: False
        - RandCropImage: # 对图像做随机裁剪
            size: 224
        - RandFlipImage: # 对图像做随机翻转
            flip_code: 1
        - NormalizeImage: # 对图像做归一化
            scale: 1./255.
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage: # 将图像从HWC格式转成CHW格式
    mix:
        - MixupOperator: # mixup数据增广，在全局配置use_mix=True时生效
            alpha: 0.2
```

PaddleClas中也包含了`AutoAugment`, `RandAugment`等数据增广方法，也可以通过在配置文件中配置，从而添加到训练过程的数据预处理中。每个数据转换的方法均以类实现，方便迁移和复用，更多的数据处理具体实现过程可以参考：`ppcls/data/imaug/operators.py`。

对于组成一个batch的数据，也可以使用mixup或者cutmix等方法进行数据增广。PaddleClas中集成了`MixupOperator`, `CutmixOperator`, `FmixOperator`等基于batch的数据增广方法，可以在配置文件中配置mix参数进行配置，更加具体的实现可以参考`ppcls/data/imaug/batch_operators.py`。

图像分类中，数据后处理主要为`argmax`操作，在此不再赘述。

* 模型结构

在配置文件中，模型结构定义如下

```yaml
ARCHITECTURE:
    name: "EfficientNetB0"
    params: # 模型需要传入的额外参数，如果没有可不填
        padding_type : "SAME"
        override_params:
            drop_connect_rate: 0.1
```


`ARCHITECTURE.name`表示模型名称，`ARCHITECTURE.params`表示需要额外传入的参数，默认为空。所有的模型名称均在`/ppcls/modeling/architectures/__init__.py`中定义。

对应的，在`tools/program.py`中，通过`create_model`方法创建模型对象。

```python
def create_model(architecture, classes_num):
    name = architecture["name"]
    params = architecture.get("params", {})
    return architectures.__dict__[name](class_dim=classes_num, **params)
```

* 损失函数

PaddleClas中，包含了`CELoss`, `MixCELoss`, `GoogLeNetLoss`, `JSDivLoss`, `MultiLabelLoss`等损失函数，均定义在`ppcls/modeling/loss.py`中。

在`tools/program.py`文件中，使用`create_loss`构建模型的损失函数，不同训练策略中所需要的损失函数与计算方法不同，PaddleClas在构建损失函数过程中，主要考虑了以下几个因素。

1. 是否使用label smooth
2. 是否使用mixup或者cutmix
3. 是否使用蒸馏方法进行训练
4. 是否进行多标签训练

```python
def create_loss(feeds,
                out,
                architecture,
                classes_num=1000,
                epsilon=None,
                use_mix=False,
                use_distillation=False,
                multilabel=False):
    if architecture["name"] == "GoogLeNet":
        assert len(out) == 3, "GoogLeNet should have 3 outputs"
        loss = GoogLeNetLoss(class_dim=classes_num, epsilon=epsilon)
        return loss(out[0], out[1], out[2], feeds["label"])

    if use_distillation:
        assert len(out) == 2, ("distillation output length must be 2, "
                               "but got {}".format(len(out)))
        loss = JSDivLoss(class_dim=classes_num, epsilon=epsilon)
        return loss(out[1], out[0])

    if use_mix:
        loss = MixCELoss(class_dim=classes_num, epsilon=epsilon)
        feed_y_a = feeds['y_a']
        feed_y_b = feeds['y_b']
        feed_lam = feeds['lam']
        return loss(out, feed_y_a, feed_y_b, feed_lam)
    else:
        if not multilabel:
            loss = CELoss(class_dim=classes_num, epsilon=epsilon)
        else:
            loss = MultiLabelLoss(class_dim=classes_num, epsilon=epsilon)
        return loss(out, feeds["label"])
```

* 优化器和学习率衰减、权重衰减策略

图像分类任务中，`Momentum`是一种比较常用的优化器，PaddleClas中提供了`Momentum`与`RMSProp`两种优化器策略。

权重衰减策略是一种比较常用的正则化方法，主要用于防止模型过拟合。PaddleClas中提供了`L1Decay`和`L2Decay`两种权重衰减策略。

学习率衰减是图像分类任务中必不可少的精度提升训练方法，PaddleClas目前支持`Cosine`, `Piecewise`, `CosineWarmup`, `ExponentialWarmup`等学习率衰减策略。

在配置文件中，优化器和权重衰减策略可以通过以下的字段进行配置。

```yaml
OPTIMIZER:
    function: 'Momentum' # Momentum优化器
    params:
        momentum: 0.9
    regularizer:
        function: 'L2' # L1 means L1Decay, L2 means L2Decay
        factor: 0.00010
```

学习率衰减策略可以通过以下的字段进行配置。

```yaml
LEARNING_RATE:
    function: 'Piecewise' # Piecewise学习率衰减策略
    params:
        lr: 0.1 # 初始学习率
        decay_epochs: [30, 60, 90] # 学习率下降时对应的epoch数量
        gamma: 0.1 # 学习率衰减倍数
```

在`tools/program.py`中使用`create_optimizer`创建优化器和学习率对象。

```python
def create_optimizer(config, parameter_list=None):
    # create learning_rate instance
    lr_config = config['LEARNING_RATE']
    lr_config['params'].update({
        'epochs': config['epochs'],
        'step_each_epoch':
        config['total_images'] // config['TRAIN']['batch_size'],
    })
    lr = LearningRateBuilder(**lr_config)()

    # create optimizer instance
    opt_config = config['OPTIMIZER']
    opt = OptimizerBuilder(**opt_config)
    return opt(lr, parameter_list), lr
 ```

 不同优化器和权重衰减策略均以类的形式实现，具体实现可以参考文件`ppcls/optimizer/optimizer.py`；不同的学习率衰减策略可以参考文件`ppcls/optimizer/learning_rate.py`。


* 训练时评估与模型存储

模型在训练的时候，可以设置模型保存的间隔，也可以选择每隔若干个epoch对验证集进行评估，从而可以保存在验证集上精度最佳的模型。配置文件中，可以通过下面的字段进行配置。

```yaml
save_interval: 1 # 模型保存的epoch间隔
validate: True # 是否进行训练时评估
valid_interval: 1 # 评估的epoch间隔
```

模型存储是通过 Paddle 框架的 `paddle.save()` 函数实现的，存储的是模型的 persistable 版本，便于继续训练。具体实现如下

```python
def save_model(net, optimizer, model_path, epoch_id, prefix='ppcls'):
    # just save model in trainer_id=0
    if paddle.distributed.get_rank() != 0:
        return
    model_path = os.path.join(model_path, str(epoch_id))
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    # save student model during distillation
    _save_student_model(net, model_prefix)

    paddle.save(net.state_dict(), model_prefix + ".pdparams")
    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    logger.info("Already save model in {}".format(model_path))
```

在保存的时候有两点需要注意：
1. 只在0号节点上保存模型。否则多卡训练的时候，如果所有节点都保存模型到相同的路径，则多个节点写文件时可能会发生写文件冲突，导致最终保存的模型无法被正确加载。
2. 优化器参数也需要存储，方便后续的加载断点进行训练。

1.2.3 预测部署代码和方式。

* 如果希望在服务端使用cpp进行部署，可以参考[cpp inference预测教程](../../../deploy/cpp_infer/readme.md)。
* 如果希望将分类模型部署为服务，可以参考[hub serving预测部署教程](../../../deploy/hubserving/readme.md)。
* 如果希望将对分类模型进行量化，可以参考[Paddle Slim量化教程](../../../deploy/slim/quant/README.md)。
* 如果希望在移动端使用分类模型进行预测，可以参考[PaddleLite预测部署教程](../../../deploy/lite/readme.md)。


## 2. 如何贡献代码

### 2.1 PaddleClas分支说明

PaddleClas未来将维护2种分支，分别为：

* release/x.x系列分支：为稳定的发行版本分支，会适时打tag发布版本，适配Paddle的release版本。当前最新的分支为release/2.0分支，是当前默认分支，适配Paddle v2.0.0。随着版本迭代，release/x.x系列分支会越来越多，默认维护最新版本的release分支，前1个版本分支会修复bug，其他的分支不再维护。
* develop分支：为开发分支，适配Paddle的develop版本，主要用于开发新功能。如果有同学需要进行二次开发，请选择develop分支。为了保证develop分支能在需要的时候拉出release/x.x分支，develop分支的代码只能使用Paddle最新release分支中有效的api。也就是说，如果Paddle develop分支中开发了新的api，但尚未出现在release分支代码中，那么请不要在PaddleClas中使用。除此之外，对于不涉及api的性能优化、参数调整、策略更新等，都可以正常进行开发。

PaddleClas的历史分支，未来将不再维护。考虑到一些同学可能仍在使用，这些分支还会继续保留：

* release/static分支：这个分支曾用于静态图的开发与测试，目前兼容>=1.7版本的Paddle。如果有特殊需求，要适配旧版本的Paddle，那还可以使用这个分支，但除了修复bug外不再更新代码。
* dygraph-dev分支：这个分支将不再维护，也不再接受新的代码，请使用的同学尽快迁移到develop分支。


PaddleClas欢迎大家向repo中积极贡献代码，下面给出一些贡献代码的基本流程。

### 2.2 PaddleClas代码提交流程与规范

#### 2.2.1 fork和clone代码

* 跳转到[PaddleClas GitHub首页](https://github.com/PaddlePaddle/PaddleClas)，然后单击 Fork 按钮，生成自己目录下的仓库，比如 `https://github.com/USERNAME/PaddleClas`。


<div align="center">
<img src="../../images/quick_start/community/001_fork.png"  width = "600" />
</div>


* 将远程仓库clone到本地

```shell
# 拉取develop分支的代码
git clone https://github.com/USERNAME/PaddleClas.git -b develop
cd PaddleClas
```

clone的地址可以从下面获取



<div align="center">
<img src="../../images/quick_start/community/002_clone.png"  width = "600" />
</div>

#### 2.2.2 和远程仓库建立连接

首先通过`git remote -v`查看当前远程仓库的信息。

```
origin    https://github.com/USERNAME/PaddleClas.git (fetch)
origin    https://github.com/USERNAME/PaddleClas.git (push)
```

只有clone的远程仓库的信息，也就是自己用户名下的 PaddleClas，接下来我们创建一个原始 PaddleClas 仓库的远程主机，命名为 upstream。

```shell
git remote add upstream https://github.com/PaddlePaddle/PaddleClas.git
```

使用`git remote -v`查看当前远程仓库的信息，输出如下，发现包括了origin和upstream 2个远程仓库。

```
origin    https://github.com/USERNAME/PaddleClas.git (fetch)
origin    https://github.com/USERNAME/PaddleClas.git (push)
upstream    https://github.com/PaddlePaddle/PaddleClas.git (fetch)
upstream    https://github.com/PaddlePaddle/PaddleClas.git (push)
```

这主要是为了后续在提交pull request(PR)时，始终保持本地仓库最新。

#### 2.2.3 创建本地分支

可以基于当前分支创建新的本地分支，命令如下。

```shell
git checkout -b new_branch
```

也可以基于远程或者上游的分支创建新的分支，命令如下。

```shell
# 基于用户远程仓库(origin)的develop创建new_branch分支
git checkout -b new_branch origin/develop
# 基于上游远程仓库(upstream)的develop创建new_branch分支
# 如果需要从upstream创建新的分支，需要首先使用git fetch upstream获取上游代码
git checkout -b new_branch upstream/develop
```

最终会显示切换到新的分支，输出信息如下

```
Branch new_branch set up to track remote branch develop from upstream.
Switched to a new branch 'new_branch'
```

#### 2.2.4 使用pre-commit勾子

Paddle 开发人员使用 pre-commit 工具来管理 Git 预提交钩子。 它可以帮助我们格式化源代码（C++，Python），在提交（commit）前自动检查一些基本事宜（如每个文件只有一个 EOL，Git 中不要添加大文件等）。

pre-commit测试是 Travis-CI 中单元测试的一部分，不满足钩子的 PR 不能被提交到 PaddleClas，首先安装并在当前目录运行它：

```shell
pip install pre-commit
pre-commit install
```

* **注意**
1. Paddle 使用 clang-format 来调整 C/C++ 源代码格式，请确保 `clang-format` 版本在 3.8 以上。
2. 通过pip install pre-commit和conda install -c conda-forge pre-commit安装的yapf稍有不同的，PaddleClas 开发人员使用的是`pip install pre-commit`。


#### 2.2.5 修改与提交代码

可以通过`git status`查看改动的文件。
对PaddleClas的`README.md`做了一些修改，希望提交上去。则可以通过以下步骤

```shell
git add README.md
pre-commit
```

重复上述步骤，直到pre-comit格式检查不报错。如下所示。

<div align="center">
<img src="../../images/quick_start/community/003_precommit_pass.png"  width = "600" />
</div>


使用下面的命令完成提交。

```shell
git commit -m "your commit info"
```

#### 2.2.6 保持本地仓库最新

获取 upstream 的最新代码并更新当前分支。这里的upstream来自于2.2节的`和远程仓库建立连接`部分。

```shell
git fetch upstream
# 如果是希望提交到其他分支，则需要从upstream的其他分支pull代码，这里是develop
git pull upstream develop
```

#### 2.2.7 push到远程仓库

```shell
git push origin new_branch
```

#### 2.2.7 提交Pull Request

点击new pull request，选择本地分支和目标分支，如下图所示。在PR的描述说明中，填写该PR所完成的功能。接下来等待review，如果有需要修改的地方，参照上述步骤更新 origin 中的对应分支即可。

<div align="center">
<img src="../../images/quick_start/community/004_create_pr.png"  width = "600" />
</div>



#### 2.2.8 签署CLA协议和通过单元测试

* 签署CLA
在首次向PaddlePaddle提交Pull Request时，您需要您签署一次CLA(Contributor License Agreement)协议，以保证您的代码可以被合入，具体签署方式如下：

1. 请您查看PR中的Check部分，找到license/cla，并点击右侧detail，进入CLA网站
2. 点击CLA网站中的“Sign in with GitHub to agree”,点击完成后将会跳转回您的Pull Request页面


#### 2.2.9 删除分支

* 删除远程分支

在 PR 被 merge 进主仓库后，我们可以在 PR 的页面删除远程仓库的分支。

也可以使用 `git push origin :分支名` 删除远程分支，如：


```shell
git push origin :new_branch
```

* 删除本地分支

```shell
# 切换到develop分支，否则无法删除当前分支
git checkout develop

# 删除new_branch分支
git branch -D new_branch
```

#### 2.2.10 提交代码的一些约定

为了使官方维护人员在评审代码时更好地专注于代码本身，请您每次提交代码时，遵守以下约定：

1）请保证Travis-CI 中单元测试能顺利通过。如果没过，说明提交的代码存在问题，官方维护人员一般不做评审。

2）提交PUll Request前：

请注意commit的数量。

原因：如果仅仅修改一个文件但提交了十几个commit，每个commit只做了少量的修改，这会给评审人带来很大困扰。评审人需要逐一查看每个commit才能知道做了哪些修改，且不排除commit之间的修改存在相互覆盖的情况。

建议：每次提交时，保持尽量少的commit，可以通过git commit --amend补充上次的commit。对已经Push到远程仓库的多个commit，可以参考[squash commits after push](https://stackoverflow.com/questions/5667884/how-to-squash-commits-in-git-after-they-have-been-pushed)。

请注意每个commit的名称：应能反映当前commit的内容，不能太随意。

3）如果解决了某个Issue的问题，请在该PUll Request的第一个评论框中加上：fix #issue_number，这样当该PUll Request被合并后，会自动关闭对应的Issue。关键词包括：close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved，请选择合适的词汇。详细可参考[Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages)。

此外，在回复评审人意见时，请您遵守以下约定：

1）官方维护人员的每一个review意见都希望得到回复，这样会更好地提升开源社区的贡献。

- 对评审意见同意且按其修改完的，给个简单的Done即可；
- 对评审意见不同意的，请给出您自己的反驳理由。

2）如果评审意见比较多,

- 请给出总体的修改情况。
- 请采用`start a review`进行回复，而非直接回复的方式。原因是每个回复都会发送一封邮件，会造成邮件灾难。


## 3. 总结

* 开源社区依赖于众多开发者与用户的贡献和反馈，在这里感谢与期待大家向PaddleClas提出宝贵的意见与pull request，希望我们可以一起打造一个领先实用全面的图像分类代码仓库！

## 4. 参考文献
1. [PaddlePaddle本地开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/08_contribution/index_cn.html)
2. [向开源框架提交pr的过程](https://blog.csdn.net/vim_wj/article/details/78300239)
