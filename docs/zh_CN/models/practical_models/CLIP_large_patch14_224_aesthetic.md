# 美观度打分模型

------


## 目录

- [1. 模型和应用场景介绍](#1)
- [2. 模型快速体验](#2)
    - [2.1 安装 paddlepaddle](#2.1)
    - [2.2 安装 paddleclas](#2.2)
- [3. 模型预测](#3)
    - [3.1 环境配置](#3.1)
    - [3.2 模型预测](#3.2)
      - [3.2.1 基于训练引擎预测](#3.2.1)
      - [3.2.2 基于推理引擎预测](#3.2.2)


<a name="1"></a>

## 1. 模型和应用场景介绍

该案例提供了用户使用 PaddleClas 的基于 CLIP_large_patch14_224 网络构建图像美观度打分的模型。该模型可以自动为图像打分，对于越符合人类审美的图像，得分越高，越不符合人类审美的图像，得分越低，可用于推荐和搜索等应用场景。本案例引用自[美观度](https://github.com/christophschuhmann/improved-aesthetic-predictor)，权重由官方权重转换而来。得分较高和得分较低的两张图片如下：
<center><img src='https://user-images.githubusercontent.com/94225063/215502324-e22b72dc-bb6a-42fa-8f9d-d1069b74c6b7.jpg' width=800></center>
可以看到，相比于右图，左图更加符合人类审美。

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

使用如下命令快速安装 paddleclas：

```  
pip3 install paddleclas
```

<a name="3"></a>

## 3. 模型预测

<a name="3.1"></a>  

### 3.1 环境配置

* 安装：请先参考文档[环境准备](../../installation.md) 配置 PaddleClas 运行环境。

<a name="3.2"></a>

### 3.2 模型预测

<a name="3.2.1"></a>

### 3.2.1 基于训练引擎预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/practical_models/CLIP_large_patch14_224_aesthetic.yaml \
    -o Arch.pretrained=True
```

输出结果如下：

```
[{'scores': Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [7.8476]), 'file_name': 'deploy/images/practical/aesthetic_score_predictor/Highscore.png'}]
```

**备注：**

* 这里`-o Arch.pretrained=True"` 指定了使用训练好的预训练权重，如果指定其他权重，只需替换对应的路径即可。

* 默认是对 `deploy/images/practical/aesthetic_score_predictor/Highscore.png` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。

<a name="3.2.2"></a>

### 3.2.2 基于推理引擎预测

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于 Paddle Inference 推理引擎的介绍，可以参考 [Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

首先，我们提供了将权重和模型转换的脚本，执行该脚本可以得到对应的 inference 模型：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/practical_models/CLIP_large_patch14_224_aesthetic.yaml \
    -o Arch.pretrained=True \
    -o Global.save_inference_dir=deploy/models/CLIP_large_patch14_224_aesthetic_infer
```
执行完该脚本后会在 `deploy/models/` 下生成 `CLIP_large_patch14_224_aesthetic_infer` 文件夹，`models` 文件夹下应有如下文件结构：

```
├── CLIP_large_patch14_224_aesthetic_infer
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
├── CLIP_large_patch14_224_aesthetic_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

得到 inference 模型之后基于推理引擎进行预测：
返回 `deploy` 目录：

```
cd ../
```

运行下面的命令，对图像 `./images/practical/aesthetic_score_predictor/Highscore.png` 进行有水印/无水印分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c ./configs/practical_models/aesthetic_score_predictor/inference_aesthetic_score_predictor.yaml
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c ./configs/practical_models/aesthetic_score_predictor/inference_aesthetic_score_predictor.yaml -o Global.use_gpu=False
```

输出结果如下。

```
Highscore.png:	score(s): [7.85]
```
