# PP-ShiTu Pipeline评估

## 目录
- [1. 模型准备](#1-模型准备)
- [2. 数据集准备](#2-数据集准备)
- [3. 模型评估](#3-模型评估)

<a name="1. 模型准备"></a>

## 1. 模型准备
创建存放模型的文件夹`deploy/models`，并下载轻量级主体检测、识别模型，命令如下：
```shell
cd deploy
mkdir models
cd models

# 下载检测模型并解压
# wget {检测模型下载链接} && tar -xf {检测模型压缩包名称}
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0_infer.tar && tar -xf general_PPLCNetV2_base_pretrained_v1.0_infer.tar

# 下载识别 inference 模型并解压
#wget {识别模型下载链接} && tar -xf {识别模型压缩包名称}
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar && tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
```

解压完成后，`models`文件夹下有如下文件结构：
```
├── inference_model_name
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
└── det_model_name
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

<a name="2. 数据集准备"></a>

## 2. 数据集准备

### 2.1 使用官方场景库数据
官方场景库介绍以及数据集下载详见[场景库应用](./application_scenarios.md)。以交通工具场景库为例，下载并解压完成后，`datasets/Vechicles`文件夹下应有如下文件结构：
```shel
├── Vechicles/
│   ├── Gallery/
│   ├── Index/
│   ├── Query/
│   ├── gallery_list.txt/
│   ├── query_list.txt/
│   └── label_list.txt/
└── ...
```
其中，`Gallery`文件夹中存放的是用于构建索引库的原始图像，`Index`表示基于原始图像构建得到的索引库信息，`Query`文件夹存放的是用于检索的图像列表，`gallery_list.txt`和`query_list.txt`分别为索引库和检索图像的标签文件，`label_list.txt`是标签的中英文对照文件。

### 2.2 准备迁移应用的数据集
如果测试模型在迁移应用的具体数据上的识别精度，需要准备对应的数据。迁移应用的具体数据集数据量根据实际情况收集，尽量避免数据过少；将收集的数据分为两部分：建库图像（gallery）和测试图像（query）。其中，建库数据数据量无需过多，但需要保证每个类别图像包含该类别物体的各种外观情况（不同角度、形状等）。

收集并划分好建库图像和测试图像后，需要生成对应的真值文件（`gallery_list.txt`和`query_list.txt`），真值文件格式如下：
```
# 每一行采用“空格”分割图像路径与标签
image_path_1 label_1
image_path_2 label_1
image_path_3 label_1
image_path_4 label_2
...
```

<a name="3. 模型评估"></a>

## 3. 模型评估
模型评估配置文件详见：`./deploy/configs/evaluation_general.yaml`

配置文件部分字段说明如下:
```
Global.det_inference_model_dir    检测模型地址
Global.rec_inference_model_dir    识别模型地址

Eval.name                         评测数据集名称
Eval.image_root                   评测数据集query图像地址
Eval.cls_label_path               评测数据集query_list.txt地址
Eval.output_dir                   评测结果保存地址

IndexProcess.image_root           评测数据集gallery图像地址
IndexProcess.index_dir            评测数据集index保存地址
IndexProcess.data_file            评测数据集gallery_list.txt地址
```

**注意**：如果使用官方场景库数据，评测数据集index保存地址`IndexProcess.index_dir`需要与官方提供的`datasets_name/Index`文件夹区分开。

以交通工具场景数据集`Vechicles`为例，运行以下命令进行模型评估：
```python
cd deploy
python python/eval_shitu_pipeline.py -c ./configs/evaluation_general.yaml \
-o Global.det_inference_model_dir=./models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer \
-o Global.rec_inference_model_dir=./models/general_PPLCNetV2_base_pretrained_v1.0_infer \
-o Eval.image_root=./datasets/Vechicles \
-o Eval.cls_label_path=./datasets/Vechicles/query_list.txt \
-o Eval.output_dir=./datasets/Vechicles/eval_out \
-o IndexProcess.image_root=./datasets/Vechicles \
-o IndexProcess.index_dir=./datasets/Vechicles/test_index \
-o IndexProcess.data_file=./datasets/Vechicles/gallery_list.txt
```

输出结果如下:
```shell
...
recal_1: 0.9400
recal_2: 0.9800
recal_3: 0.9900
recal_4: 1.0000
recal_5: 1.0000
```
