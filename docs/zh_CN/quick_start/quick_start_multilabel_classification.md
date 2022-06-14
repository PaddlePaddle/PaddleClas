# 多标签分类 quick start

基于 [NUS-WIDE-SCENE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) 数据集，体验多标签分类的训练、评估、预测的过程，该数据集是 NUS-WIDE 数据集的一个子集。请首先安装 PaddlePaddle 和 PaddleClas，具体安装步骤可详看 [环境准备](../installation/install_paddleclas.md)。


## 目录

* [1. 数据和模型准备](#1)
* [2. 模型训练](#2)
* [3. 模型评估](#3)
* [4. 模型预测](#4)
* [5. 基于预测引擎预测](#5)
  * [5.1 导出 inference model](#5.1)
  * [5.2 基于预测引擎预测](#5.2)

<a name="1"></a>
## 1. 数据和模型准备

* 进入 `PaddleClas` 目录。

```
cd path_to_PaddleClas
```

* 创建并进入 `dataset/NUS-WIDE-SCENE` 目录，下载并解压 NUS-WIDE-SCENE 数据集。

```shell
mkdir dataset/NUS-WIDE-SCENE
cd dataset/NUS-WIDE-SCENE
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/NUS-SCENE-dataset.tar
tar -xf NUS-SCENE-dataset.tar
```

* 返回 `PaddleClas` 根目录

```
cd ../../
```

<a name="2"></a>
## 2. 模型训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml
```

训练 10 epoch 之后，验证集最好的正确率应该在 0.95 左右。

<a name="3"></a>

## 3. 模型评估

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml \
    -o Arch.pretrained="./output/MobileNetV1/best_model"
```

<a name="4"></a>
## 4. 模型预测

```bash
python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml \
    -o Arch.pretrained="./output/MobileNetV1/best_model"
```

得到类似下面的输出：
```  
[{'class_ids': [6, 13, 17, 23, 26, 30], 'scores': [0.95683, 0.5567, 0.55211, 0.99088, 0.5943, 0.78767], 'file_name': './deploy/images/0517_2715693311.jpg', 'label_names': []}]
```

<a name="5"></a>
## 5. 基于预测引擎预测

<a name="5.1"></a>
### 5.1 导出 inference model

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml \
    -o Arch.pretrained="./output/MobileNetV1/best_model"
```
inference model 的路径默认在当前路径下 `./inference`

<a name="5.2"></a>
### 5.2 基于预测引擎预测

首先进入 `deploy` 目录下：

```bash
cd ./deploy
```

通过预测引擎推理预测：

```
python3 python/predict_cls.py \
     -c configs/inference_multilabel_cls.yaml
```

得到类似下面的输出：
```
0517_2715693311.jpg:    class id(s): [6, 13, 17, 23, 26, 30], score(s): [0.96, 0.56, 0.55, 0.99, 0.59, 0.79], label_name(s): []
```
