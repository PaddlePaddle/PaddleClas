# ResNet 系列
-----
## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
    - [1.3 Benchmark](#1.3)
      - [1.3.1 基于 V100 GPU 的预测速度](#1.3.1)
      - [1.3.2 基于 T4 GPU 的预测速度](#1.3.2)
- [2. 模型快速体验](#2)
    - [2.1 安装 paddlepaddle](#2.1)
    - [2.2 安装 paddleclas](#2.2)
    - [2.3 预测](#2.3)
- [3. 模型训练、评估和预测](#3)
    - [3.1 环境配置](#3.1)
    - [3.2 数据准备](#3.2)
    - [3.3 模型训练](#3.3)
    - [3.4 模型评估](#3.4)
    - [3.5 模型预测](#3.5)
- [4. 模型推理部署](#4)
  - [4.1 推理模型准备](#4.1)
    - [4.1.1 基于训练得到的权重导出 inference 模型](#4.1.1)
    - [4.1.2 直接下载 inference 模型](#4.1.2)
  - [4.2 基于 Python 预测引擎推理](#4.2)
    - [4.2.1 预测单张图像](#4.2.1)
    - [4.2.2 基于文件夹的批量预测](#4.2.2)
  - [4.3 基于 C++ 预测引擎推理](#4.3)
  - [4.4 服务化部署](#4.4)
  - [4.5 端侧部署](#4.5)
  - [4.6 Paddle2ONNX 模型转换与预测](#4.6)

<a name='1'></a>

## 1. 模型介绍

<a name='1.1'></a>

### 1.1 模型简介

ResNet 系列模型是在 2015 年提出的，一举在 ILSVRC2015 比赛中取得冠军，top5 错误率为 3.57%。该网络创新性的提出了残差结构，通过堆叠多个残差结构从而构建了 ResNet 网络。实验表明使用残差块可以有效地提升收敛速度和精度。

斯坦福大学的 Joyce Xu 将 ResNet 称为「真正重新定义了我们看待神经网络的方式」的三大架构之一。由于 ResNet 卓越的性能，越来越多的来自学术界和工业界学者和工程师对其结构进行了改进，比较出名的有 Wide-ResNet, ResNet-vc, ResNet-vd, Res2Net 等，其中 ResNet-vc 与 ResNet-vd 的参数量和计算量与 ResNet 几乎一致，所以在此我们将其与 ResNet 统一归为 ResNet 系列。

PaddleClas 提供的 ResNet 系列的模型包括 ResNet50，ResNet50_vd，ResNet50_vd_ssld，ResNet200_vd 等 16 个预训练模型。在训练层面上，ResNet 的模型采用了训练 ImageNet 的标准训练流程，而其余改进版模型采用了更多的训练策略，如 learning rate 的下降方式采用了 cosine decay，引入了 label smoothing 的标签正则方式，在数据预处理加入了 mixup 的操作，迭代总轮数从 120 个 epoch 增加到 200 个 epoch。

其中，后缀使用`_ssld`的模型采用了 SSLD 知识蒸馏，保证模型结构不变的情况下，进一步提升了模型的精度。


<a name='1.2'></a>

### 1.2 模型指标

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNet18         | 0.710           | 0.899           | 0.696                    | 0.891                    | 3.660     | 11.690    |
| ResNet18_vd      | 0.723           | 0.908           |                          |                          | 4.140     | 11.710    |
| ResNet34         | 0.746           | 0.921           | 0.732                    | 0.913                    | 7.360     | 21.800    |
| ResNet34_vd      | 0.760           | 0.930           |                          |                          | 7.390     | 21.820    |
| ResNet34_vd_ssld      | 0.797           | 0.949           |                          |                          | 7.390     | 21.820    |
| ResNet50         | 0.765           | 0.930           | 0.760                    | 0.930                    | 8.190     | 25.560    |
| ResNet50_vc      | 0.784           | 0.940           |                          |                          | 8.670     | 25.580    |
| ResNet50_vd      | 0.791           | 0.944           | 0.792                    | 0.946                    | 8.670     | 25.580    |
| ResNet101        | 0.776           | 0.936           | 0.776                    | 0.938                    | 15.520    | 44.550    |
| ResNet101_vd     | 0.802           | 0.950           |                          |                          | 16.100    | 44.570    |
| ResNet152        | 0.783           | 0.940           | 0.778                    | 0.938                    | 23.050    | 60.190    |
| ResNet152_vd     | 0.806           | 0.953           |                          |                          | 23.530    | 60.210    |
| ResNet200_vd     | 0.809           | 0.953           |                          |                          | 30.530    | 74.740    |
| ResNet50_vd_ssld | 0.830           | 0.964           |                          |                          | 8.670     | 25.580    |
| Fix_ResNet50_vd_ssld | 0.840           | 0.970           |                          |                          | 17.696     | 25.580    |
| ResNet101_vd_ssld | 0.837           | 0.967           |                          |                          | 16.100    | 44.570     |

**备注：** `Fix_ResNet50_vd_ssld` 是固定 `ResNet50_vd_ssld` 除 FC 层外所有的网络参数，在 320x320 的图像输入分辨率下，基于 ImageNet-1k 数据集微调得到。


<a name='1.3'></a>

## 1.3 Benchmark

<a name='1.3.1'></a>

### 1.3.1 基于 V100 GPU 的预测速度

| Models                 | Size | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
|:--:|:--:|:--:|:--:|:--:|
| ResNet18         | 224       | 1.22               | 2.19               | 3.63               |
| ResNet18_vd      | 224       | 1.26               | 2.28               | 3.89               |
| ResNet34         | 224       | 1.97               | 3.25               | 5.70               |
| ResNet34_vd      | 224       | 2.00               | 3.28               | 5.84               |
| ResNet34_vd_ssld      | 224  | 2.00               | 3.26               | 5.85               |
| ResNet50         | 224       |  2.54               | 4.79               | 7.40               |
| ResNet50_vc      | 224       | 2.57               | 4.83               | 7.52               |
| ResNet50_vd      | 224       |  2.60               | 4.86               | 7.63               |
| ResNet101        | 224       |  4.37               | 8.18               | 12.38              |
| ResNet101_vd     | 224       |  4.43               | 8.25               | 12.60              |
| ResNet152        | 224       | 6.05               | 11.41              | 17.33              |
| ResNet152_vd     | 224       |  6.11               | 11.51              | 17.59              |
| ResNet200_vd     | 224       |  7.70               | 14.57              | 22.16              |
| ResNet50_vd_ssld | 224       | 2.59           | 4.87               | 7.62               |
| ResNet101_vd_ssld  | 224     | 4.43             | 8.25             | 12.58            |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT。

<a name='1.3.2'></a>

### 1.3.2 基于 T4 GPU 的预测速度

| Models            | Size | Latency(ms)<br>FP16<br>bs=1 | Latency(ms)<br>FP16<br>bs=4 | Latency(ms)<br>FP16<br>bs=8 | Latency(ms)<br>FP32<br>bs=1 | Latency(ms)<br>FP32<br>bs=4 | Latency(ms)<br>FP32<br>bs=8 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNet18          | 224       | 1.3568                       | 2.5225                       | 3.61904                      | 1.45606                      | 3.56305                      | 6.28798                      |
| ResNet18_vd       | 224       | 1.39593                      | 2.69063                      | 3.88267                      | 1.54557                      | 3.85363                      | 6.88121                      |
| ResNet34          | 224       | 2.23092                      | 4.10205                      | 5.54904                      | 2.34957                      | 5.89821                      | 10.73451                     |
| ResNet34_vd       | 224       | 2.23992                      | 4.22246                      | 5.79534                      | 2.43427                      | 6.22257                      | 11.44906                     |
| ResNet34_vd_ssld       | 224       | 2.23992                      | 4.22246                      | 5.79534                      | 2.43427                      | 6.22257                      | 11.44906                     |
| ResNet50          | 224       | 2.63824                      | 4.63802                      | 7.02444                      | 3.47712                      | 7.84421                      | 13.90633                     |
| ResNet50_vc       | 224       | 2.67064                      | 4.72372                      | 7.17204                      | 3.52346                      | 8.10725                      | 14.45577                     |
| ResNet50_vd       | 224       | 2.65164                      | 4.84109                      | 7.46225                      | 3.53131                      | 8.09057                      | 14.45965                     |
| ResNet101         | 224       | 5.04037                      | 7.73673                      | 10.8936                      | 6.07125                      | 13.40573                     | 24.3597                      |
| ResNet101_vd      | 224       | 5.05972                      | 7.83685                      | 11.34235                     | 6.11704                      | 13.76222                     | 25.11071                     |
| ResNet152         | 224       | 7.28665                      | 10.62001                     | 14.90317                     | 8.50198                      | 19.17073                     | 35.78384                     |
| ResNet152_vd      | 224       | 7.29127                      | 10.86137                     | 15.32444                     | 8.54376                      | 19.52157                     | 36.64445                     |
| ResNet200_vd      | 224       | 9.36026                      | 13.5474                      | 19.0725                      | 10.80619                     | 25.01731                     | 48.81399                     |
| ResNet50_vd_ssld  | 224       | 2.65164                      | 4.84109                      | 7.46225                      | 3.53131                      | 8.09057                      | 14.45965                     |
| Fix_ResNet50_vd_ssld  | 320       | 3.42818                      | 7.51534                      | 13.19370                      | 5.07696                      | 14.64218                      | 27.01453                     |
| ResNet101_vd_ssld | 224       | 5.05972                      | 7.83685                      | 11.34235                     | 6.11704                      | 13.76222                     | 25.11071                     |

**备注：** 推理过程使用 TensorRT。

<a name="2"></a>   
    
## 2. 模型快速体验

<a name="2.1"></a>   
    
### 2.1 安装 paddlepaddle

- 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

- 您的机器是CPU，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

<a name="2.2"></a>  

### 2.2 安装 paddleclas

使用如下命令快速安装 paddleclas

```  
pip3 install paddleclas
```
<a name="2.3"></a> 
    
### 2.3 预测

* 在命令行中使用 ResNet50 的权重快速预测
    
```bash
paddleclas --model_name=ResNet50  --infer_imgs="docs/images/inference_deployment/whl_demo.jpg"
```
    
结果如下：
```
>>> result
class_ids: [8, 7, 86, 82, 80], scores: [0.97968, 0.02028, 3e-05, 1e-05, 0.0], label_names: ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'black grouse'], filename: docs/images/inference_deployment/whl_demo.jpg
Predict complete!
```
    
**备注**： 更换 ResNet 的其他 scale 的模型时，只需替换 `model_name`，如将此时的模型改为 `ResNet18` 时，只需要将 `--model_name=ResNet50` 改为 `--model_name=ResNet18` 即可。   

    
* 在 Python 代码中预测
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = 'docs/images/inference_deployment/whl_demo.jpg'
result = clas.predict(infer_imgs)
print(next(result))
```

**备注**：`PaddleClas.predict()` 为可迭代对象（`generator`），因此需要使用 `next()` 函数或 `for` 循环对其迭
代调用。每次调用将以 `batch_size` 为单位进行一次预测，并返回预测结果。返回结果示例如下：

```
>>> result
[{'class_ids': [8, 7, 86, 82, 80], 'scores': [0.97968, 0.02028, 3e-05, 1e-05, 0.0], 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'black grouse'], 'filename': 'docs/images/inference_deployment/whl_demo.jpg'}]
```


<a name="3"></a> 
    
## 3. 模型训练、评估和预测
    
<a name="3.1"></a>  

### 3.1 环境配置

* 安装：请先参考 [Paddle 安装教程](../installation/install_paddle.md) 以及 [PaddleClas 安装教程](../installation/install_paddleclas.md) 配置 PaddleClas 运行环境。

<a name="3.2"></a> 

### 3.2 数据准备

请在[ImageNet 官网](https://www.image-net.org/)准备 ImageNet-1k 相关的数据。


进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，将下载好的数据命名为 `ILSVRC2012` ，存放于此。 `ILSVRC2012` 目录中具有以下数据：

```
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
├── train_list.txt
...
├── val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
├── val_list.txt
```

其中 `train/` 和 `val/` 分别为训练集和验证集。`train_list.txt` 和 `val_list.txt` 分别为训练集和验证集的标签文件。
    
**备注：** 

* 关于 `train_list.txt`、`val_list.txt`的格式说明，可以参考[PaddleClas分类数据集格式说明](../data_preparation/classification_dataset.md#1-数据集格式说明) 。


<a name="3.3"></a> 

### 3.3 模型训练 


在 `ppcls/configs/ImageNet/ResNet/ResNet50.yaml` 中提供了 ResNet50 训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml 
```


**备注：** 

* 当前精度最佳的模型会保存在 `output/ResNet50/best_model.pdparams`

<a name="3.4"></a>

### 3.4 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
    -o Global.pretrained_model=output/ResNet50/best_model
```

其中 `-o Global.pretrained_model="output/ResNet50/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

<a name="3.5"></a>

### 3.5 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
    -o Global.pretrained_model=output/ResNet50/best_model 
```

输出结果如下：

```
[{'class_ids': [8, 7, 86, 82, 80], 'scores': [0.97968, 0.02028, 3e-05, 1e-05, 0.0], 'file_name': 'docs/images/inference_deployment/whl_demo.jpg', 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'black grouse']}]
```

**备注：** 

* 这里`-o Global.pretrained_model="output/ResNet50/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。
    
* 默认是对 `docs/images/inference_deployment/whl_demo.jpg` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。
    
* 默认输出的是 Top-5 的值，如果希望输出 Top-k 的值，可以指定`-o Infer.PostProcess.topk=k`，其中，`k` 为您指定的值。


    
<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a> 

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。
    
当使用 Paddle Inference 推理时，加载的模型类型为 inference 模型。本案例提供了两种获得 inference 模型的方法，如果希望得到和文档相同的结果，请选择[直接下载 inference 模型](#6.1.2)的方式。

    
<a name="4.1.1"></a> 

### 4.1.1 基于训练得到的权重导出 inference 模型

此处，我们提供了将权重和模型转换的脚本，执行该脚本可以得到对应的 inference 模型：

```bash
python3 tools/export_model.py \
    -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
    -o Global.pretrained_model=output/ResNet50/best_model \
    -o Global.save_inference_dir=deploy/models/ResNet50_infer
```
执行完该脚本后会在 `deploy/models/` 下生成 `ResNet50_infer` 文件夹，`models` 文件夹下应有如下文件结构：

```
├── ResNet50_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```


<a name="4.1.2"></a> 

### 4.1.2 直接下载 inference 模型

[4.1.1 小节](#4.1.1)提供了导出 inference 模型的方法，此处也提供了该场景可以下载的 inference 模型，可以直接下载体验。

```
cd deploy/models
# 下载 inference 模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar && tar -xf ResNet50_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── ResNet50_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="4.2"></a> 

### 4.2 基于 Python 预测引擎推理


<a name="4.2.1"></a>  

#### 4.2.1 预测单张图像

返回 `deploy` 目录：

```
cd ../
```

运行下面的命令，对图像 `./images/ImageNet/ILSVRC2012_val_00000010.jpeg` 进行分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/ResNet50_infer
# 使用下面的命令使用 CPU 进行预测
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/ResNet50_infer -o Global.use_gpu=False
```

输出结果如下。

```
ILSVRC2012_val_00000010.jpeg:	class id(s): [153, 332, 229, 204, 265], score(s): [0.41, 0.39, 0.05, 0.04, 0.04], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'Angora, Angora rabbit', 'Old English sheepdog, bobtail', 'Lhasa, Lhasa apso', 'toy poodle']
```

<a name="4.2.2"></a>  

#### 4.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/ResNet50_infer -o Global.infer_imgs=images/ImageNet/
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
ILSVRC2012_val_00000010.jpeg:	class id(s): [153, 332, 229, 204, 265], score(s): [0.41, 0.39, 0.05, 0.04, 0.04], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'Angora, Angora rabbit', 'Old English sheepdog, bobtail', 'Lhasa, Lhasa apso', 'toy poodle']
ILSVRC2012_val_00010010.jpeg:	class id(s): [902, 626, 531, 487, 761], score(s): [0.47, 0.10, 0.05, 0.04, 0.03], label_name(s): ['whistle', 'lighter, light, igniter, ignitor', 'digital watch', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'remote control, remote']
ILSVRC2012_val_00020010.jpeg:	class id(s): [178, 211, 246, 236, 210], score(s): [1.00, 0.00, 0.00, 0.00, 0.00], label_name(s): ['Weimaraner', 'vizsla, Hungarian pointer', 'Great Dane', 'Doberman, Doberman pinscher', 'German short-haired pointer']
ILSVRC2012_val_00030010.jpeg:	class id(s): [80, 23, 83, 93, 136], score(s): [1.00, 0.00, 0.00, 0.00, 0.00], label_name(s): ['black grouse', 'vulture', 'prairie chicken, prairie grouse, prairie fowl', 'hornbill', 'European gallinule, Porphyrio porphyrio']
```


<a name="4.3"></a> 

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../inference_deployment/cpp_deploy.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../inference_deployment/cpp_deploy_on_windows.md)完成相应的预测库编译和模型预测工作。

<a name="4.4"></a> 

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。
    
PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../inference_deployment/paddle_serving_deploy.md)来完成相应的部署工作。

<a name="4.5"></a> 

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。
    
PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../inference_deployment/paddle_lite_deploy.md)来完成相应的部署工作。

<a name="4.6"></a> 

### 4.6 Paddle2ONNX 模型转换与预测
    
Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](@shuilong)来完成相应的部署工作。
