# 图像识别快速开始

本文档包含3个部分：环境配置、图像识别体验、未知类别的图像识别体验。

如果图像类别已经存在于图像索引库中，那么可以直接参考[图像识别体验](#图像识别体验)章节，完成图像识别过程；如果希望识别未知类别的图像，即图像类别之前不存在于索引库中，那么可以参考[未知类别的图像识别体验](#未知类别的图像识别体验)章节，完成建立索引并识别的过程。

## 目录

* [1. 环境配置](#环境配置)
* [2. 图像识别体验](#图像识别体验)
  * [2.1 下载、解压inference 模型与demo数据](#下载、解压inference_模型与demo数据)
  * [2.2 Logo识别与检索](#Logo识别与检索)
    * [2.2.1 识别单张图像](#识别单张图像)
    * [2.2.2 基于文件夹的批量识别](#基于文件夹的批量识别)
* [3. 未知类别的图像识别体验](#未知类别的图像识别体验)
  * [3.1 基于自己的数据集构建索引库](#基于自己的数据集构建索引库)
  * [3.2 基于新的索引库的图像识别](#基于新的索引库的图像识别)


<a name="环境配置"></a>
## 1. 环境配置

* 安装：请先参考[快速安装](./installation.md)配置PaddleClas运行环境。

* 进入`deploy`运行目录。本部分所有内容与命令均需要在`deploy`目录下运行，可以通过下面的命令进入`deploy`目录。

  ```
  cd deploy
  ```

<a name="图像识别体验"></a>
## 2. 图像识别体验

检测模型与4个方向(Logo、动漫人物、车辆、商品)的识别inference模型、测试数据下载地址以及对应的配置文件地址如下。

| 模型简介       | 推荐场景   | 测试数据地址  | inference模型  | 预测配置文件  | 构建索引库的配置文件 |
| ------------  | ------------- | ------- | -------- | ------- | -------- |
| 通用主体检测模型 | 通用场景  | -  |[模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar) | - | - |
| Logo识别模型 | Logo场景  | [数据下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/logo_demo_data_v1.0.tar) |  [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/logo_rec_ResNet50_Logo3K_v1.0_infer.tar) | [inference_logo.yaml](../../../deploy/configs/inference_logo.yaml) | [build_logo.yaml](../../../deploy/configs/build_logo.yaml) |
| 动漫人物识别模型 | 动漫人物场景  | [数据下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/cartoon_demo_data_v1.0.tar) | [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/cartoon_rec_ResNet50_iCartoon_v1.0_infer.tar) | [inference_cartoon.yaml](../../../deploy/configs/inference_cartoon.yaml) | [build_cartoon.yaml](../../../deploy/configs/build_cartoon.yaml) |
| 车辆细分类模型 | 车辆场景  | [数据下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/vehicle_demo_data_v1.0.tar) |  [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/vehicle_cls_ResNet50_CompCars_v1.0_infer.tar) | [inference_vehicle.yaml](../../../deploy/configs/inference_vehicle.yaml) | [build_vehicle.yaml](../../../deploy/configs/build_vehicle.yaml) |
| 商品识别模型 | 商品场景  | [数据下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/product_demo_data_v1.0.tar) |  [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/product_ResNet50_vd_Inshop_v1.0_infer.tar) | [inference_inshop.yaml](../../../deploy/configs/) | [build_inshop.yaml](../../../deploy/configs/build_inshop.yaml) |


**注意**：windows 环境下如果没有安装wget,下载模型时可将链接复制到浏览器中下载，并解压放置在相应目录下；linux或者macOS用户可以右键点击，然后复制下载链接，即可通过`wget`命令下载。


* 可以按照下面的命令下载并解压数据与模型

```shell
mkdir dataset
cd dataset
# 下载demo数据并解压
wget {数据下载链接地址} && tar -xf {压缩包的名称}
cd ..

mkdir models
cd models
# 下载识别inference模型并解压
wget {模型下载链接地址} && tar -xf {压缩包的名称}
cd ..
```


<a name="下载、解压inference_模型与demo数据"></a>
### 2.1 下载、解压inference 模型与demo数据

Logo识别为例，下载通用检测、识别模型以及Logo识别demo数据，命令如下。

```shell
mkdir models
cd models
# 下载通用检测inference模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar && tar -xf ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
# 下载识别inference模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/logo_rec_ResNet50_Logo3K_v1.0_infer.tar && tar -xf logo_rec_ResNet50_Logo3K_v1.0_infer.tar

cd ..
mkdir dataset
cd dataset
# 下载demo数据并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/logo_demo_data_v1.0.tar && tar -xf logo_demo_data_v1.0.tar
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

其中`data_file.txt`是用于构建索引库的图像列表文件，`gallery`文件夹中是所有用于构建索引库的图像原始文件，`index`文件夹中是构建索引库生成的索引文件，`query`是用来测试识别效果的demo图像。

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

<a name="Logo识别与检索"></a>
### 2.2 Logo识别与检索

以Logo识别demo为例，展示识别与检索过程（如果希望尝试其他方向的识别与检索效果，在下载解压好对应的demo数据与模型之后，替换对应的配置文件即可完成预测）。


<a name="识别单张图像"></a>
#### 2.2.1 识别单张图像

运行下面的命令，对图像`./dataset/logo_demo_data_v1.0/query/logo_auxx-1.jpg`进行识别与检索

```shell
python3.7 python/predict_system.py -c configs/inference_logo.yaml
```

待检索图像如下所示。
<div align="center">
<img src="../../images/recognition/logo_demo/query/logo_auxx-1.jpg"  width = "400" />
</div>


最终输出结果如下。

```
[{'bbox': [129, 219, 230, 253], 'rec_docs': ['auxx-2', 'auxx-1', 'auxx-2', 'auxx-1', 'auxx-2'], 'rec_scores': array([3.09635019, 3.09635019, 2.83965826, 2.83965826, 2.64057827])}]
```

其中bbox表示检测出的主体所在位置，rec_docs表示索引库中与检出主体最相近的若干张图像对应的标签，rec_scores表示对应的相似度。


匹配的一些示例图像如下所示。

<center class="half">
    <img src="../../images/recognition/logo_demo/gallery/auxx-1_59_0.jpg" width="100"/><img src="../../images/recognition/logo_demo/gallery/auxx-1_79_0.jpg" width="100"/><img src="../../images/recognition/logo_demo/gallery/auxx-2_47_0.jpg" width="100"/><img src="../../images/recognition/logo_demo/gallery/auxx-2_59_0.jpg" width="100"/><img src="../../images/recognition/logo_demo/gallery/auxx-2_79_0.jpg" width="100"/>
</center>

<a name="基于文件夹的批量识别"></a>
#### 2.2.2 基于文件夹的批量识别

如果希望预测文件夹内的图像，可以直接修改配置文件中的`Global.infer_imgs`字段，也可以通过下面的`-o`参数修改对应的配置。

```shell
python3.7 python/predict_system.py -c configs/inference_logo.yaml -o Global.infer_imgs="./dataset/logo_demo_data_v1.0/query"
```

更多地，可以通过修改`Global.rec_inference_model_dir`字段来更改识别inference模型的路径，通过修改`IndexProcess.index_path`字段来更改索引库索引的路径。


<a name="未知类别的图像识别体验"></a>
## 3. 未知类别的图像识别体验

对图像`xxxx`进行识别，命令如下

```shell
python3.7 python/predict_system.py -c configs/inference_logo.yaml -o Global.infer_imgs="./dataset/logo_demo_data_v1.0/query/new_img.jpg"
```

输出结果如下

```
old index out
```

由于索引库中不包含对应的索引信息，所以这里的识别结果有误，此时我们可以通过构建新的索引库的方式，完成未知类别的图像识别。


当索引库中的图像无法覆盖我们实际识别的场景时，即在预测未知类别的图像时，我们需要将对应类别的相似图像添加到索引库中，从而完成对未知类别的图像识别，这一过程是不需要重新训练的。

<a name="基于自己的数据集构建索引库"></a>
### 3.1 基于自己的数据集构建索引库

首先需要获取待入库的原始图像文件(保存在`./dataset/logo_demo_data_v1.0/gallery`文件夹中)以及对应的标签信息，记录原始图像文件的文件名与标签信息）保存在文本文件[data_file_update.txt](./dataset/logo_demo_data_v1.0/data_file_update.txt)中）。

然后使用下面的命令构建index索引，加速识别后的检索过程。

```shell
python3.7 python/build_gallery.py -c configs/build_logo.yaml -o IndexProcess.data_file="./dataset/logo_demo_data_v1.0/data_file_update.txt" -o IndexProcess.index_path="./dataset/logo_demo_data_v1.0/index_update"
```

最终新的索引信息保存在文件夹`./dataset/logo_demo_data_v1.0/index_update`中。使用新的索引库对上述索引


<a name="基于新的索引库的图像识别"></a>
### 3.2 基于新的索引库的图像识别

对图像`xxxx`进行识别，运行命令如下。

```shell
python3.7 python/predict_system.py -c configs/inference_logo.yaml -o Global.infer_imgs="./dataset/logo_demo_data_v1.0/query/new_img.jpg" -o IndexProcess.index_path="./dataset/logo_demo_data_v1.0/index_update"
```

输出结果如下。

```
new index out
```

识别结果无误。
