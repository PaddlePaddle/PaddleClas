# 图像分类模型自动压缩示例

目录：
- [图像分类模型自动压缩示例](#图像分类模型自动压缩示例)
  - [1. 简介](#1-简介)
  - [2. Benchmark](#2-benchmark)
    - [PaddleClas模型](#paddleclas模型)
  - [3. 自动压缩流程](#3-自动压缩流程)
      - [3.1 准备环境](#31-准备环境)
      - [3.2 准备数据集](#32-准备数据集)
      - [3.3 准备预测模型](#33-准备预测模型)
      - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
  - [4.预测部署](#4预测部署)
      - [4.1 Paddle Inference 验证性能](#41-paddle-inference-验证性能)
      - [4.2 PaddleLite端侧部署](#42-paddlelite端侧部署)
  - [5.FAQ](#5faq)


## 1. 简介
本示例将以图像分类模型MobileNetV1为例，介绍如何使用PaddleClas中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为量化训练和蒸馏。

## 2. Benchmark

### PaddleClas模型

|           模型           | 策略 | Top-1 Acc | GPU 耗时(ms) | ARM CPU 耗时(ms) | 配置文件 | Inference模型 |
|:----------------------:|:------:|:---------:|:----------:|:--------------:|:------:|:-----:|
| MobileNetV3_small_x1_0 | Baseline |   68.19   |     -      |                | - | [Model](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_small_x1_0_infer.tar) |
| MobileNetV3_small_x1_0 | 量化+蒸馏 |   64.90   |     -      |                | [Config](./configs/MobileNetV3_small_x1_0/qat_dis.yaml) ||
|        ResNet50        | Baseline |  76.46       |            |       -        | - | [Model](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar) |
|      ResNet50_vd       | 量化+蒸馏 |   76.08   |            |       -        | [Config](./configs/ResNet50_vd/qat_dis.yaml) |  |



- ARM CPU 测试环境：`SDM865(4xA77+4xA55)`
- Nvidia GPU 测试环境：
  - 硬件：NVIDIA Tesla T4 单卡
  - 软件：CUDA 11.2, cuDNN 8.0, TensorRT 8.4
  - 测试配置：batch_size: 1, image size: 224

## 3. 自动压缩流程

#### 3.1 准备环境

- python >= 3.6
- PaddlePaddle >= 2.5 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.5

安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle==2.5.1
# GPU 以Ubuntu、CUDA 11.2为例
python -m pip install paddlepaddle-gpu==2.5.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

安装paddleslim：
```shell
pip install paddleslim
```

若使用`run_ppclas.py`脚本，需安装paddleclas：
```shell
git clone https://github.com/PaddlePaddle/PaddleClas.git -b release/2.5
cd PaddleClas
pip install --upgrade -r requirements.txt
```

#### 3.2 准备数据集
本案例默认以ImageNet1k数据进行自动压缩实验，如数据集为非ImageNet1k格式数据， 请参考[PaddleClas数据准备文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/data_preparation/classification_dataset.md)。将下载好的数据集放在当前目录下`./ILSVRC2012`。


#### 3.3 准备预测模型
预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

可在[PaddleClas预训练模型库](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)中直接获取Inference模型，具体可参考下方获取MobileNetV1模型示例：

```shell
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar
tar -xf ResNet50_infer.tar
```
也可根据[PaddleClas文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/inference_deployment/export_model.md)导出Inference模型。

#### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口 ```paddleslim.auto_compression.AutoCompression``` 对模型进行量化训练和蒸馏。配置config文件中模型路径、数据集路径、蒸馏、量化和训练等部分的参数，配置完成后便可开始自动压缩。

**单卡启动**

```shell
export CUDA_VISIBLE_DEVICES=0
python run_ppclas.py \
        --compression_config_path='./configs/ResNet50/qat_dis.yaml' \
        --reader_config_path='./configs/ResNet50/data_reader.yaml' \
        --save_dir='./save_quant_ResNet50/'
```

**多卡启动**

图像分类训练任务中往往包含大量训练数据，以ImageNet为例，ImageNet22k数据集中包含1400W张图像，如果使用单卡训练，会非常耗时，使用分布式训练可以达到几乎线性的加速比。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch run.py --save_dir='./save_quant_resnet50/' --config_path='./configs/ResNet50/qat_dis.yaml'
```
多卡训练指的是将训练任务按照一定方法拆分到多个训练节点完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。服务节点在收到所有训练节点传来的梯度后，会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。多卡训练一轮训练能训练```batch size * num gpus```的数据，比如单卡的```batch size```为32，单轮训练的数据量即32，而四卡训练的```batch size```为32，单轮训练的数据量为128。

注意：

- 参数设置：```learning rate``` 与 ```batch size``` 呈线性关系，这里单卡 ```batch size``` 为32，对应的 ```learning rate``` 为0.015，那么如果 ```batch size``` 减小4倍改为8，```learning rate``` 也需除以4；多卡时 ```batch size``` 为32，```learning rate``` 需乘上卡数。所以改变 ```batch size``` 或改变训练卡数都需要对应修改 ```learning rate```。

- 如需要使用`PaddleClas`中的数据预处理和`DataLoader`，可以使用`run_ppclas.py`脚本启动，启动方式跟以上示例相同，但配置需要对其```PaddleClas```，可参考[ViT配置文件](./configs/VIT/data_reader.yml)。



## 4.预测部署

#### 4.1 Paddle Inference 验证性能

量化模型在GPU上可以使用TensorRT进行加速，在CPU上可以使用MKLDNN进行加速。

以下字段用于配置预测参数：

| 参数名 | 含义 |
|:------:|:------:|
| model_path | inference 模型文件所在目录，该目录下需要有文件 .pdmodel 和 .pdiparams 两个文件 |
| model_filename | inference_model_dir文件夹下的模型文件名称 |
| params_filename | inference_model_dir文件夹下的参数文件名称 |
| data_path | 数据集路径  |
| batch_size | 预测一个batch的大小   |
| image_size | 输入图像的大小   |
| use_gpu | 是否使用 GPU 预测   |
| use_trt | 是否使用 TesorRT 预测引擎   |
| use_mkldnn | 是否启用```MKL-DNN```加速库，注意```use_mkldnn```与```use_gpu```同时为```True```时，将忽略```use_mkldnn```，而使用```GPU```预测  |
| cpu_num_threads | CPU预测时，使用CPU线程数量，默认10  |
| use_fp16 | 使用TensorRT时，是否启用```FP16```  |
| use_int8 | 是否启用```INT8``` |

注意：
- 请注意模型的输入数据尺寸，如InceptionV3输入尺寸为299，部分模型需要修改参数：```image_size```


- TensorRT预测：

环境配置：如果使用 TesorRT 预测引擎，需安装的是带有TensorRT的PaddlePaddle，使用以下指令查看本地cuda版本，并且在[下载链接](https://www.paddlepaddle.org.cn/inference/user_guides/download_lib.html#python)中下载对应cuda版本和对应python版本的PaddlePaddle安装包。

    ```shell
    cat /usr/local/cuda/version.txt ### CUDA Version 10.2.89
    ### 10.2.89 为cuda版本号，可以根据这个版本号选择需要安装的带有TensorRT的PaddlePaddle安装包。
    ```

```shell
python test_ppclas.py \
      --model_path=./save_quant_resnet50 \
      --use_trt=True \
      --use_int8=True \
      --use_gpu=True \
      --data_path=./dataset/ILSVRC2012/
```

- MKLDNN预测：

```shell
python test_ppclas \
      --model_path=./save_quant_resnet50 \
      --data_path=./dataset/ILSVRC2012/ \
      --cpu_num_threads=10 \
      --use_mkldnn=True \
      --use_int8=True
```
