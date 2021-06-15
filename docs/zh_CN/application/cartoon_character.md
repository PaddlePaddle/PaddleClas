# 动漫人物识别
## 简介
   自七十年代以来，人脸识别已经成为了计算机视觉和生物识别领域研究最多的主题之一。近年来，传统的人脸识别方法已经被基于卷积神经网络（CNN）的深度学习方法代替。目前，人脸识别技术广泛应用于安防、商业、金融、智慧自助终端、娱乐等各个领域。而在行业应用强烈需求的推动下，动漫媒体越来越受到关注，动漫人物的人脸识别也成为一个新的研究领域。

## 数据集
近日，来自爱奇艺的一项新研究提出了一个新的基准数据集，名为iCartoonFace。该数据集由 5013 个动漫角色的 389678 张图像组成，并带有 ID、边界框、姿势和其他辅助属性。 iCartoonFace 是目前图像识别领域规模最大的卡通媒体数据集，而且质量高、注释丰富、内容全面，其中包含相似图像、有遮挡的图像以及外观有变化的图像。
与其他数据集相比，iCartoonFace无论在图像数量还是实体数量上，均具有明显领先的优势:

![icartoon](./icartoon1.jpg)

论文地址：https://arxiv.org/pdf/1907.1339

## 推理
检索任务的推理过程主要分为两步： 1. 建库;   2. 检索。  通过配置文件，我们可以实现自动建库，并对配置文件里指定的图像进行检索，返回识别结果

**1. 获取数据**
```
cd dataset
wget http://10.9.189.15:8088/metric_learning_dygraph/0607/PaddleClas/dataset/icartoon.tar.gz
tar -xvf icartoon.tar.gz
```

**2. 获取模型**

**3. 修改配置**

**4. 前向推理**
```
python deploy/python/predict_system.py -c  deploy/configs/inference_icartoon.yaml
```

## 训练
**单卡训练**
```
Python tools/train.py -c ppcls/configs/Cartoon/ResNet50_icartoon.yaml

```
**多卡训练**
```
python -m paddle.distributed.launch \
        --gpus="0,1,2,3" \
        tools/train.py \
        -c ./ppcls/configs/Cartoonface/ResNet50_icartoon.yaml 

```
**评估**
```
Python tools/eval.py -c ppcls/configs/Cartoon/ResNet50_icartoon.yaml

