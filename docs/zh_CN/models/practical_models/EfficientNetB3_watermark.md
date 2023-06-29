# 有水印/无水印分类模型

------


## 目录

- [1. 模型和应用场景介绍](#1)
- [2. 模型快速体验](#2)
    - [2.1 安装 paddlepaddle](#2.1)
    - [2.2 安装 paddleclas](#2.2)
- [3. 模型预测](#3)
    - [3.1 模型预测](#3.1)
      - [3.1.1 基于训练引擎预测](#3.1.1)
      - [3.1.2 基于推理引擎预测](#3.1.2)


<a name="1"></a>

## 1. 模型和应用场景介绍

该案例提供了用户使用 PaddleClas 的基于 EfficientNetB3 网络构建有水印/无水印（这里的水印包括数字照片上留下的一些logo、信息、网址等）的分类模型。该模型可以广泛应用于审核场景、海量数据过滤场景等。本案例引用自[水印识别](https://github.com/LAION-AI/LAION-5B-WatermarkDetection),权重由官方权重转换而来。具体有无水印的图片对比如下：
<center><img src='https://user-images.githubusercontent.com/94225063/212879681-f115d6f8-85c8-4cda-a07e-5f5b00d8236a.jpeg' width=800></center>
可以看到，左图中有水印，右图中无水印。

<a name="2"></a>

## 2. 模型快速体验

<a name="2.1"></a>  

### 2.1 安装 paddlepaddle

- 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

- 您的机器是 CPU，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

<a name="2.2"></a>

### 2.2 安装 paddleclas

请确保已clone本项目，本地构建安装：

```
cd path/to/PaddleClas
#使用下面的命令构建
pip install -v -e .
```

<a name="3"></a>

## 3. 模型预测

<a name="3.1"></a>

### 3.1 模型预测

<a name="3.1.1"></a>

### 3.1.1 基于训练引擎预测

加载预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/practical_models/EfficientNetB3_watermark.yaml
```

输出结果如下：

```
[{'class_ids': [0], 'scores': [0.9653296619653702], 'label_names': ['contains_watermark'], 'file_name': 'deploy/images/practical/watermark_exists/watermark_example.png'}]
```

**备注：**

* 默认是对 `deploy/images/practical/watermark_exists/watermark_example.png` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。

* 二分类默认的阈值为0.5， 如果需要指定阈值，可以重写 `Infer.PostProcess.threshold` 。

<a name="3.1.2"></a>

### 3.1.2 基于推理引擎预测

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于 Paddle Inference 推理引擎的介绍，可以参考 [Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

首先，我们提供了将权重和模型转换的脚本，执行该脚本可以得到对应的 inference 模型：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/practical_models/EfficientNetB3_watermark.yaml \
    -o Global.save_inference_dir=deploy/models/EfficientNetB3_watermark_infer
```
执行完该脚本后会在 `deploy/models/` 下生成 `EfficientNetB3_watermark_infer` 文件夹，`models` 文件夹下应有如下文件结构：

```
├── EfficientNetB3_watermark_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

当然，也可以选择直接下载的方式：

```
cd deploy/models
# 下载 inference 模型并解压
wget https://paddleclas.bj.bcebos.com/models/practical/inference/EfficientNetB3_watermark_infer.tar && tar -xf EfficientNetB3_watermark_infer.tar
```
解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── EfficientNetB3_watermark_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

得到 inference 模型之后基于推理引擎进行预测：
返回 `deploy` 目录：

```
cd ../
```

运行下面的命令，对图像 `./images/practical/watermark_exists/watermark_example.png` 进行有水印/无水印分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3 python/predict_cls.py -c ./configs/practical_models/watermark_exists/inference_watermark_exists.yaml
# 使用下面的命令使用 CPU 进行预测
python3 python/predict_cls.py -c ./configs/practical_models/watermark_exists/inference_watermark_exists.yaml -o Global.use_gpu=False
```

输出结果如下。

```
watermark_example.png:	class id(s): [0], score(s): [0.97], label_name(s): ['contains_watermark']
```

**备注：** 二分类默认的阈值为0.5， 如果需要指定阈值，可以重写 `PostProcess.ThreshOutput.threshold` 。
