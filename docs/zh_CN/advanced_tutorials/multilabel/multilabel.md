# 多标签分类quick start

基于[NUS-WIDE-SCENE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)数据集，体验多标签分类的训练、评估、预测的过程，该数据集是NUS-WIDE数据集的一个子集。请事先参考[安装指南](install.md)配置运行环境和克隆PaddleClas代码。

## 一、数据和模型准备

* 进入PaddleClas目录。

```
cd path_to_PaddleClas
```

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

## 二、环境准备

### 2.1 下载预训练模型

本例展示基于ResNet50_vd模型的多标签分类流程，因此首先下载ResNet50_vd的预训练模型

```bash
mkdir pretrained
cd pretrained
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
cd ../
```

## 三、模型训练

```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch \
    --gpus="0" \
    tools/train.py \
        -c ./configs/quick_start/ResNet50_vd_multilabel.yaml
```

训练10epoch之后，验证集最好的正确率应该在0.72左右。

## 四、模型评估

```bash
python tools/eval.py \
    -c ./configs/quick_start/ResNet50_vd_multilabel.yaml \
    -o pretrained_model="./output/ResNet50_vd/best_model/ppcls" \
    -o load_static_weights=False
```

评估指标采用mAP，验证集的mAP应该在0.57左右。

## 五、模型预测

```bash
python tools/infer/infer.py \
    -i "./dataset/NUS-WIDE-SCENE/NUS-SCENE-dataset/images/0199_434752251.jpg" \
    --model ResNet50_vd \
    --pretrained_model "./output/ResNet50_vd/best_model/ppcls" \
    --use_gpu True \
    --load_static_weights False \
    --multilabel True \
    --class_num 33
```

得到类似下面的输出：
```    
    class id: 3, probability: 0.6025
    class id: 23, probability: 0.5491
    class id: 32, probability: 0.7006
```