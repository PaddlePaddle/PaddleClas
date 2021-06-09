# 图像识别快速开始

图像识别主要包含3个部分：主体检测得到检测框、识别提取特征、根据特征进行检索。

## 1. 环境配置

请先参考[快速安装](./installation.md)配置PaddleClas运行环境。

本部分内容需要在`deploy`文件夹下运行，在PaddleClas代码的根目录下，可以通过以下方法进入该文件夹

```shell
cd deploy
```


## 2. inference 模型和数据下载

检测模型与4个方向(Logo、动漫人物、车辆、商品)的识别inference模型以及测试数据下载方法如下。。

| 模型简介       | 推荐场景   | 测试数据地址  | inference模型 |
| ------------  | ------------- | ------- | -------- |
| 通用主体检测模型 | 通用场景  | -  |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar) |
| Logo识别模型 | Logo场景  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/logo_demo_data_v1.0.tar) |  [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/logo_rec_ResNet50_Logo3K_v1.0_infer.tar) |
| 动漫人物识别模型 | 动漫人物场景  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/cartoon_demo_data_v1.0.tar) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/cartoon_rec_ResNet50_iCartoon_v1.0_infer.tar) |
| 车辆细分类模型 | 车辆场景  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/vehicle_demo_data_v1.0.tar) |  [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/vehicle_cls_ResNet50_CompCars_v1.0_infer.tar) |
| 商品识别模型 | 商品场景  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/product_demo_data_v1.0.tar) |  [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/product_ResNet50_vd_Inshop_v1.0_infer.tar) |


**注意**：windows 环境下如果没有安装wget,下载模型时可将链接复制到浏览器中下载，并解压放置在相应目录下


* 下载并解压数据与模型

```shell
mkdir dataset
cd dataset
# 下载demo数据并解压
wget {url/of/data} && tar -xf {name/of/data/package}
cd ..

mkdir models
cd models
# 下载识别inference模型并解压
wget {url/of/inference model} && tar -xf {name/of/inference model/package}
cd ..
```


### 2.1 下载通用检测模型

```shell
mkdir models
cd models
# 下载通用检测inference模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar && tar -xf ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
cd ..
```


### 2.1 Logo识别

以Logo识别demo为例，按照下面的命令下载demo数据与模型。

```shell
mkdir dataset
cd dataset
# 下载demo数据并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/logo_demo_data_v1.0.tar && tar -xf logo_demo_data_v1.0.tar
cd ..

mkdir models
cd models
# 下载识别inference模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/logo_rec_ResNet50_Logo3K_v1.0_infer.tar && tar -xf logo_rec_ResNet50_Logo3K_v1.0_infer.tar
cd ..
```

解压完毕后，`dataset`文件夹下应有如下文件结构：

```
├── logo_demo_data_v1.0
│   ├── data_file.txt
│   ├── gallery
│   ├── index
│   └── query
├── ...
```

`models`文件夹下应有如下文件结构：

```
├── logo_rec_ResNet50_Logo3K_v1.0_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── ppyolov2_r50vd_dcn_mainbody_v1.0_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

按照下面的方式可以完成对于图片的检索

```shell
python3.7 python/predict_system.py -c configs/inference_logo.yaml
```

配置文件中，部分关键字段解释如下

```yaml
Global:
  infer_imgs: "./dataset/logo_demo_data_v1.0/query/" # 预测图像
  det_inference_model_dir: "./models/logo_rec_ResNet50_Logo3K_v1.0_infer/" # 检测inference模型文件夹
  rec_inference_model_dir: "./models/ppyolov2_r50vd_dcn_mainbody_v1.0_infer/" # 识别inference模型文件夹
  batch_size: 1 # 预测的批大小
  image_shape: [3, 640, 640] # 检测的图像尺寸
  threshold: 0.5 # 检测的阈值，得分超过该阈值的检测框才会被检出
  max_det_results: 1 # 用于图像识别的检测框数量，符合阈值条件的检测框中，根据得分，最多对其中的max_det_results个检测框做后续的识别

# indexing engine config
IndexProcess:
  index_path: "./dataset/logo_demo_data_v1.0/index/" # 索引文件夹，用于识别特征提取之后的索引
  search_budget: 100
  return_k: 5 # 从底库中反馈return_k个数量的最相似内容
  dist_type: "IP"
```



最终输出结果如下

```
[{'bbox': [25, 21, 483, 382], 'rec_docs': ['AKG', 'AKG', 'AKG', 'AKG', 'AKG'], 'rec_scores': array([2.32288337, 2.31903863, 2.28398442, 2.16804123, 2.10190272])}]
```

其中bbox表示检测出的主体所在位置，rec_docs表示底库中与检出主体最相近的若干张图像对应的标签，rec_scores表示对应的相似度。

如果希望预测文件夹内的图像，可以直接修改配置文件，也可以通过下面的`-o`参数修改对应的配置。

```shell
python3.7 python/predict_system.py -c configs/inference_logo.yaml -o Global.infer_imgs="./dataset/logo_demo_data_v1.0/query"
```

如果希望在底库中新增图像，重新构建idnex，可以使用下面的命令重新构建index。

```shell
python3.7 python/build_gallery.py -c configs/build_logo.yaml
```

其中index相关配置如下。

```yaml
# indexing engine config
IndexProcess:
  index_path: "./dataset/logo_demo_data_v1.0/index/" # 保存的索引地址
  image_root: "./dataset/logo_demo_data_v1.0/" # 图像的根目录
  data_file:  "./dataset/logo_demo_data_v1.0/data_file.txt" # 图像的数据list文本，每一行包含图像的文件名与标签信息
  delimiter: "\t"
  dist_type: "IP"
  pq_size: 100
  embedding_size: 512 # 特征维度
```

需要改动的内容为：
1. 在图像根目录下面添加对应的图像内容（也可以在其子文件夹下面，保证最终根目录与数据list文本中添加的文件名合并之后，图像存在即可）
2. 图像的数据list文本中添加图像新的内容，每行包含图像文件名以及对应的标签信息。


### 2.2 其他任务的识别

如果希望尝试其他方向的识别与检索效果，在下载解压好对应的demo数据与模型之后，替换对应的配置文件即可完成预测。


| 场景   | 预测配置文件  | 构建底库的配置文件 |
| ---- | ----- | ----- |
| 动漫人物 | [inference_cartoon.yaml](../../../deploy/configs/inference_cartoon.yaml) | [build_cartoon.yaml](../../../deploy/configs/build_cartoon.yaml) |
| 车辆 | [inference_vehicle.yaml](../../../deploy/configs/inference_vehicle.yaml) | [build_vehicle.yaml](../../../deploy/configs/build_vehicle.yaml) |
| 商品 | [inference_inshop.yaml](../../../deploy/configs/) | [build_inshop.yaml](../../../deploy/configs/build_inshop.yaml) |
