# ML-Decoder多标签分类

## 目录

* [1. 模型介绍](#1)
* [2. 数据和模型准备](#2)
* [3. 模型训练](#3)
* [4. 模型评估](#4)
* [5. 模型预测](#5)
* [6. 基于预测引擎预测](#6)
  * [6.1 导出 inference model](#6.1)
* [7. 引用](#7)

<a name="1"></a>
## 1. 模型介绍

ML-Decoder是一种新的基于注意力的分类头，它通过查询来预测类别标签的存在，并且比全局平均池化能够更好地利用空间数据。ML-Decoder的特点有以下几点：

1. ML-Decoder重新设计了解码器的架构，并且使用了一种新颖的分组解码方案，使得ML-Decoder具有高效性和可扩展性，可以很好地处理数千个类别的分类任务。
2. ML-Decoder具有一致性的速度-准确度权衡，相比于使用更大的主干网络，ML-Decoder能够提供更好的性能。
3. ML-Decoder也具有多功能性，它可以作为各种分类头的替代方案，并且在使用词语查询时能够泛化到未见过的类别。通过使用新颖的查询增强方法，ML-Decoder的泛化能力进一步提高。

使用ML-Decoder，作者在多个分类任务上取得了最先进的结果：
1. 在MS-COCO多标签分类上，达到了91.4%的mAP；
2. 在NUS-WIDE零样本分类上，达到了31.1%的ZSL mAP；
3. 在ImageNet单标签分类上，使用普通的ResNet50主干网络，达到了80.7%的新高分，而没有使用额外的数据或蒸馏。

`PaddleClas` 目前支持在单标签分类和多标签分类任务中使用ML-Decoder, 且使用方式非常简单方便，须在配置文件的Arch作用域下设置use_ml_decoder=True 并给出对ML-Decoder的参数配置即可:
```yaml
# model architecture
Arch:
  name: ResNet50
  class_num: 80
  pretrained: True
  # use ml-decoder head to replace avg_pool and fc
  use_ml_decoder: True

# ml-decoder head
MLDecoder:
  class_num: 80
  query_num: 80 # default: 80, query_num <= class_num
  in_chans: 2048
```
开发者可以自行尝试使用不同的主干模型来结合ML-Decoder。

目前使用ResNet50结合ML-Decoder在COCO2017的多标签分类任务上的性能指标如下：

|        Model         | Backbone  | Resolution |  mAP  |                                                                                                                                                                                                         Links                                                                                                                                                                                                         |
|:--------------------:|:---------:|:----------:|:-----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ResNet50_ml_decoder  | ResNet50  |  224x224   | 0.795 | [config](ResNet50_ml_decoder_224.yaml) \  [model](https://bj.bcebos.com/v1/ai-studio-online/0af3af5c12b543fa9461a015bdb99aa6327439cd284e4ebea8255e5ce970a460?responseContentDisposition=attachment%3B%20filename%3DResNet50_ml_decoder_0.795.pdparams&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-09-06T11%3A17%3A51Z%2F-1%2F%2F7d78e695070117e615bb0114422ed25ec9c62df2e62bba7b039c3e33f92f3359) |
| ResNet101_ml_decoder | ResNet101 |  448x448   |       |                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ResNet152_ml_decoder | ResNet152 |  448x448   |       |                                                                                                                                                                                                                                                                                                                                                                                                                       |


基于 [COCO2017](https://cocodataset.org/) 数据集，如下将介绍添加ml-decoder进行多标签分类的训练、评估、预测的过程。请首先安装 PaddlePaddle 和 PaddleClas，具体安装步骤可详看 [环境准备](../installation.md)。



<a name="2"></a>
## 2. 数据和模型准备

* 进入 `PaddleClas` 目录。

```
cd path_to_PaddleClas
```

* 创建并进入 `dataset/COCO2017` 目录，下载并解压 COCO2017 数据集。

```shell
mkdir dataset/COCO2017
cd dataset/COCO2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip -d .
unzip val2017.zip -d .
unzip annotations_trainval2017.zip -d .
rm train2017.zip
rm val2017.zip
rm annotations_trainval2017.zip
```

* 返回 `PaddleClas` 根目录


```shell
cd ../../
# 转换训练集并生成`COCO2017_labels.txt`
python3 ./ppcls/utils/create_coco_multilabel_lists.py \
        --dataset_dir dataset/COCO2017 \
        --image_dir train2017 \
        --anno_path annotations/instances_train2017.json \
        --save_name multilabel_train_list --save_label_name
# 转换测试集
python3 ./ppcls/utils/create_coco_multilabel_lists.py \
        --dataset_dir dataset/COCO2017 \
        --image_dir val2017 \
        --anno_path annotations/instances_val2017.json \
        --save_name multilabel_val_list
```

<a name="3"></a>
## 3. 模型训练

```shell
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet50_ml_decoder.yaml
# 单卡
python3 tools/train.py \
        -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet50_ml_decoder.yaml
```

**注意:**
1. 目前`ResNet50_ml_decoder.yaml`的训练默认单卡，如进行多卡训练请相应的调节学习率。
2. 目前多标签分类的损失函数默认使用`MultiLabelAsymmetricLoss`。
2. 目前多标签分类的评估指标默认使用`MultiLabelMAP(integral)`。

<a name="4"></a>

## 4. 模型评估

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet50_ml_decoder.yaml \
    -o Arch.pretrained="./output/ResNet50_ml_decoder/best_model"
```

<a name="5"></a>
## 5. 模型预测

```bash
python3 tools/infer.py \
    -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet50_ml_decoder.yaml \
    -o Arch.pretrained="./output/ResNet50_ml_decoder/best_model"
```

得到类似下面的输出：
```
[{'class_ids': [0, 2, 7, 24, 25, 26, 33, 56], 'scores': [0.99998, 0.52104, 0.51953, 0.59292, 0.64329, 0.63605, 0.99994, 0.7054], 'label_names': ['person', 'car', 'truck', 'backpack', 'umbrella', 'handbag', 'kite', 'chair'], 'file_name': 'deploy/images/coco_000000570688.jpg'}]
```

<a name="5"></a>
## 6. 基于预测引擎预测

<a name="5.1"></a>
### 6.1 导出 inference model

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet50_ml_decoder.yaml \
    -o Arch.pretrained="./output/ResNet50_ml_decoder/best_model"
```
inference model 的路径默认在当前路径下 `./inference`

<a name="7"></a>
## 7. 引用
```
@misc{ridnik2021mldecoder,
      title={ML-Decoder: Scalable and Versatile Classification Head}, 
      author={Tal Ridnik and Gilad Sharir and Avi Ben-Cohen and Emanuel Ben-Baruch and Asaf Noy},
      year={2021},
      eprint={2111.12933},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```