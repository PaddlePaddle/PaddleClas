## 图像识别快速体验

本文档包含 2 个部分：PP-ShiTu android端 demo 快速体验与PP-ShiTu PC端 demo 快速体验。

如果图像类别已经存在于图像索引库中，那么可以直接参考[图像识别体验](#图像识别体验)章节，完成图像识别过程；如果希望识别未知类别的图像，即图像类别之前不存在于索引库中，那么可以参考[未知类别的图像识别体验](#未知类别的图像识别体验)章节，完成建立索引并识别的过程。

## 目录

- [1. PP-ShiTu android demo 快速体验](#1-pp-shitu-android-demo-快速体验)
  - [1.1 安装 PP-ShiTu android demo](#11-安装-pp-shitu-android-demo)
  - [1.2 操作说明](#12-操作说明)
- [2. PP-ShiTu PC端 demo 快速体验](#2-pp-shitu-pc端-demo-快速体验)
  - [2.1 环境配置](#21-环境配置)
  - [2.2 图像识别体验](#22-图像识别体验)
    - [2.2.1 下载、解压 demo 数据](#221-下载解压-demo-数据)
    - [2.2.2 瓶装饮料识别与检索](#222-瓶装饮料识别与检索)
      - [2.2.2.1 识别单张图像](#2221-识别单张图像)
      - [2.2.2.2 基于文件夹的批量识别](#2222-基于文件夹的批量识别)
  - [2.3 未知类别的图像识别体验](#23-未知类别的图像识别体验)
    - [2.3.1 准备新的数据与标签](#231-准备新的数据与标签)
    - [2.3.2 建立新的索引库](#232-建立新的索引库)
    - [2.3.3 基于新的索引库的图像识别](#233-基于新的索引库的图像识别)
  - [2.4 服务端识别模型列表](#24-服务端识别模型列表)

<a name="PP-ShiTu android 快速体验"></a>

## 1. PP-ShiTu android demo 快速体验

如需基于android demo项目进行二次开发，请点击：[shitu_android_demo](../../../deploy/shitu_android_demo/docs/README.md)

<a name="安装"></a>

### 1.1 安装 PP-ShiTu android demo

可以通过扫描二维码或者[点击链接](https://paddle-imagenet-models-name.bj.bcebos.com/demos/PP-ShiTu.apk)下载并安装APP

<div align=center><img src="../../images/quick_start/android_demo/PPShiTu_qrcode.png" height="400" width="400"/></div>

<a name="功能体验"></a>

### 1.2 功能体验
目前 PP-ShiTu android demo 具有图像检索、图像加库、保存检索库、初始化检索库、查看检索库标签等基本功能，接下来介绍如何体验这几个功能。

#### （1）识别图像中的物体
点击下方的“拍照识别”按钮<img src="../../images/quick_start/android_demo/paizhaoshibie_100.png" width="25" height="25"/>或者“本地识别”按钮<img src="../../images/quick_start/android_demo/bendishibie_100.png" width="25" height="25"/>，即可拍摄一张图像或者选中一张图像，然后等待几秒钟，APP便会将图像中的主体框标注出来并且在图像下方给出预测的类别以及预测时间等信息。

在选择好要检索的图片之后，首先会通过检测模型进行主体检测，得到图像中的物体的区域，然后将这块区域裁剪出来输入到识别模型中，得到对应的特征向量并在检索库中检索，返回并显示最终的检索结果。

假设待检索的图像如下：

<img src="../../images/recognition/drink_data_demo/test_images/nongfu_spring.jpeg" width="400" height="600"/>

得到的检索结果可视化如下：

<img src="../../images/quick_start/android_demo/android_nongfu_spring.JPG" width="400" height="800"/>

#### （2）向检索库中添加新的类别或物体
点击上方的“拍照上传”按钮<img src="../../images/quick_start/android_demo/paizhaoshangchuan_100.png" width="25" height="25"/>或者“本地上传”按钮<img src="../../images/quick_start/android_demo/bendishangchuan_100.png" width="25" height="25"/>，即可拍摄一张图像或从图库中选择一张图像，然后再输入这张图像的类别名字（比如`keyboard`），点击“确定”按钮，即可将图片对应的特征向量与标签加入检索库。

在选择好要入库的图片之后，首先会通过检测模型进行主体检测，得到图像中的物体的区域，然后将这块区域裁剪出来输入到识别模型中，得到对应的特征向量，再与用户输入的图像标签一起加入到检索库中。

**温馨提示：** 使用安卓demo管理类别主要用于功能体验，如果您有较为重要的数据要生成检索库，推荐使用 [检索库管理工具](../deployment/PP-ShiTu/gallery_manager.md)

#### (3) 保存检索库
点击上方的“保存修改”按钮<img src="../../images/quick_start/android_demo/baocunxiugai_100.png" width="25" height="25"/>，即可将当前库以 `latest` 的库名保存下来。
再次打开程序时，将会自动选择使用`latest`库。app仅存在一个自定义库，每次保存时会覆盖之前的库。

#### (4) 检索库恢复出厂设置
**警告：本操作无法撤销，初始化后自定义的标签和类别都会被删除，请谨慎操作**

点击上方的“初始化 ”按钮<img src="../../images/quick_start/android_demo/reset_100.png" width="25" height="25"/>，删除所有自定义的标签和类别，恢复出厂特征库。

初始化库时会删掉`latest`库（如果存在），自动将检索库和标签库切换成 `original.index` 和 `original.txt`。不管是否有保存过，自定义的标签和类别都会被清空。

#### (5) 查看当前检索库中的类别列表
点击“类别查询”按钮<img src="../../images/quick_start/android_demo/leibiechaxun_100.png" width="25" height="25"/>，即可在弹窗中查看。

当检索标签库过多（如本demo自带的196类检索标签库）时，可在弹窗中滑动查看。


## 2. PP-ShiTu PC端 demo 快速体验

<a name="环境配置"></a>

### 2.1 环境配置

* **[推荐]** 直接 pip 安装：

```bash
pip3 install paddleclas
```

* 如需使用 PaddleClas develop 分支体验最新功能，或是需要基于 PaddleClas 进行二次开发，请本地构建安装：

```bash
python3 setup.py install
```

<a name="图像识别体验"></a>

### 2.2 图像识别体验

轻量级通用主体检测模型与轻量级通用识别模型和配置文件下载方式如下表所示。

<a name="轻量级通用主体检测模型与轻量级通用识别模型"></a>

| 模型简介                | 推荐场景  | inference 模型 | 预测配置文件  |
| ---------------------- | -------- | -----------  | ------------ |
| 轻量级通用主体检测模型 | 通用场景 | [tar 格式下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar) \| [zip 格式下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.zip)                     | -                                                                        |
| 轻量级通用识别模型     | 通用场景 | [tar 格式下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0_infer.tar) \| [zip 格式下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0_infer.zip) | [inference_general.yaml](../../../deploy/configs/inference_general.yaml) |

注意：由于部分解压缩软件在解压上述 `tar` 格式文件时存在问题，建议非命令行用户下载 `zip` 格式文件并解压。`tar` 格式文件建议使用命令 `tar -xf xxx.tar` 解压。

本章节 demo 数据下载地址如下: [drink_dataset_v2.0.tar(瓶装饮料数据)](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v2.0.tar)，

下面以 **drink_dataset_v2.0.tar** 为例介绍PC端的 PP-ShiTu 快速体验流程。用户也可以自行下载并解压其它场景的数据进行体验：[22种场景数据下载](../deployment/PP-ShiTu/application_scenarios.md#1-应用场景介绍)。

如果希望体验服务端主体检测和各垂类方向的识别模型，可以参考 [2.4 服务端识别模型列表](#24-服务端识别模型列表)

**注意**

- windows 环境下如果没有安装 wget, 可以按照下面的步骤安装 wget 与 tar 命令，也可以在下载模型时将链接复制到浏览器中下载，并解压放置在相应目录下； linux 或者 macOS 用户可以右键点击，然后复制下载链接，即可通过 `wget` 命令下载。
- 如果 macOS 环境下没有安装 `wget` 命令，可以运行下面的命令进行安装。
    ```shell
    # 安装 homebrew
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)";
    # 安装 wget
    brew install wget
    ```
- 如果希望在 windows 环境下安装 wget，可以参考：[链接](https://www.cnblogs.com/jeshy/p/10518062.html)；如果希望在 windows 环境中安装 tar 命令，可以参考：[链接](https://www.cnblogs.com/chooperman/p/14190107.html)。

<a name="2.2.1"></a>

#### 2.2.1 下载、解压 demo 数据

下载 demo 数据集以及轻量级主体检测、识别模型，命令如下。

```shell
# 下载 demo 数据并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v2.0.tar && tar -xf drink_dataset_v2.0.tar
```

解压完毕后，`drink_dataset_v2.0/` 文件夹下应有如下文件结构：

```log
├── drink_dataset_v2.0/
│   ├── gallery/
│   ├── index/
│   ├── index_all/
│   └── test_images/
├── ...
```

其中 `gallery` 文件夹中存放的是用于构建索引库的原始图像，`index` 表示基于原始图像构建得到的索引库信息，`test_images` 文件夹中存放的是用于测试识别效果的图像列表。

**注意**

如果使用服务端通用识别模型，Demo 数据需要重新提取特征、够建索引，方式如下：

**在命令行中构建索引库**
```shell
paddleclas --build_gallery=True --model_name="PP-ShiTuV2" \
-o IndexProcess.image_root=./drink_dataset_v2.0/gallery/ \
-o IndexProcess.index_dir=./drink_dataset_v2.0/index \
-o IndexProcess.data_file=./drink_dataset_v2.0/gallery/drink_label.txt
```
其中参数`build_gallery(bool)`控制是否使用索引库构建模式，默认为`False`。

同时可以通过`-o`指令更改构建索引库使用的配置，字段说明如下：

- IndexProcess.image_root(str): 构建索引库使用的`gallery`图像地址。
- IndexProcess.index_dir(str): 索引库存放地址。
- IndexProcess.data_file(str): 构建索引库图像的真值文件。



<a name="瓶装饮料识别与检索"></a>

#### 2.2.2 瓶装饮料识别与检索

以瓶装饮料识别 demo 为例，展示识别与检索过程（如果希望尝试其他方向的识别与检索效果，在下载解压好对应的 demo 数据之后，替换命令中对应的文件地址即可）。


<a name="识别单张图像"></a>

##### 2.2.2.1 识别单张图像

运行下面的命令，对图像 `./drink_dataset_v2.0/test_images/100.jpeg` 进行识别与检索

待检索图像如下所示

![](../../images/recognition/drink_data_demo/test_images/100.jpeg)

**在命令行中进行识别和检索**
```shell
paddleclas --model_name=PP-ShiTuV2 --predict_type=shitu \
-o Global.infer_imgs='./drink_dataset_v2.0/test_images/100.jpeg' \
-o IndexProcess.index_dir='./drink_dataset_v2.0/index'
```
其中参数`model_name`为用于检索和识别的模型、`predict_type`设置为'shitu'模式。

同时可以通过`-o`指令更改检索图像以及索引库，字段说明如下：
- Global.infer_imgs(str)：待检索图像地址。
- IndexProcess.index_dir(str): 索引库存放地址。

最终输出结果如下：
```
[{'bbox': [437, 71, 660, 728], 'rec_docs': '元气森林', 'rec_scores': 0.7740249}, {'bbox': [221, 72, 449, 701], 'rec_docs': '元气森林', 'rec_scores': 0.6950992}, {'bbox': [794, 104, 979, 652], 'rec_docs': '元气森林', 'rec_scores': 0.6305153}], filename: ./drink_dataset_v2.0/test_images/100.jpeg
```


其中 `bbox` 表示检测出的主体所在位置，`rec_docs` 表示索引库中与检测框最为相似的类别，`rec_scores` 表示对应的置信度。


<a name="基于文件夹的批量识别"></a>

##### 2.2.2.2 基于文件夹的批量识别

如果希望预测文件夹内的图像，可以直接修改命令行中的 `-o` 参数对应的`Global.infer_imgs` 字段配置。

**在命令行中进行识别和检索**
```shell
paddleclas --model_name=PP-ShiTuV2 --predict_type=shitu \
-o Global.infer_imgs='./drink_dataset_v2.0/test_images' \
-o IndexProcess.index_dir='./drink_dataset_v2.0/index'
```

终端中会输出该文件夹内所有图像的识别结果，如下所示。

```log
...
[{'bbox': [0, 0, 600, 600], 'rec_docs': '红牛-强化型', 'rec_scores': 0.74081033}], filename: ./drink_dataset_v2.0/test_images/001.jpeg
[{'bbox': [0, 0, 514, 436], 'rec_docs': '康师傅矿物质水', 'rec_scores': 0.6918598}], filename: ./drink_dataset_v2.0/test_images/002.jpeg
[{'bbox': [138, 40, 573, 1198], 'rec_docs': '乐虎功能饮料', 'rec_scores': 0.68214047}], filename: ./drink_dataset_v2.0/test_images/003.jpeg
[{'bbox': [328, 7, 467, 272], 'rec_docs': '脉动', 'rec_scores': 0.60406065}], filename: ./drink_dataset_v2.0/test_images/004.jpeg
[{'bbox': [242, 82, 498, 726], 'rec_docs': '味全_每日C', 'rec_scores': 0.5428652}], filename: ./drink_dataset_v2.0/test_images/005.jpeg
[{'bbox': [437, 71, 660, 728], 'rec_docs': '元气森林', 'rec_scores': 0.7740249}, {'bbox': [221, 72, 449, 701], 'rec_docs': '元气森林', 'rec_scores': 0.6950992}, {'bbox': [794, 104, 979, 652], 'rec_docs': '元气森林', 'rec_scores': 0.6305153}], filename: ./drink_dataset_v2.0/test_images/100.jpeg
...
```


<a name="未知类别的图像识别体验"></a>

### 2.3 未知类别的图像识别体验

对图像 `./drink_dataset_v2.0/test_images/mosilian.jpeg` 进行识别

待检索图像如下

![](../../images/recognition/drink_data_demo/test_images/mosilian.jpeg)

执行如下识别命令

**在命令行中进行识别和检索**
```shell
paddleclas --model_name=PP-ShiTuV2 --predict_type=shitu \
-o Global.infer_imgs='./drink_dataset_v2.0/test_images/mosilian.jpeg' \
-o IndexProcess.index_dir='./drink_dataset_v2.0/index'
```

可以发现输出结果为空

由于默认的索引库中不包含对应的索引信息，所以这里的识别结果有误，此时我们可以通过构建新的索引库的方式，完成未知类别的图像识别。

当索引库中的图像无法覆盖我们实际识别的场景时，即识别未知类别的图像前，我们需要将该未知类别的相似图像（至少一张）添加到索引库中，从而完成对未知类别的图像识别。这一过程不需要重新训练模型，以识别 `mosilian.jpeg` 为例，只需按以下步骤重新构建新的索引库即可。

<a name="准备新的数据与标签"></a>

#### 2.3.1 准备新的数据与标签

首先需要将与待检索图像相似的图像列表拷贝到索引库原始图像的文件夹中。这里 PaddleClas 已经将所有的图像数据都放在文件夹 `drink_dataset_v2.0/gallery/` 中。

然后需要编辑记录了图像路径和标签信息的文本文件，这里 PaddleClas 将更新后的标签信息文件放在了 `drink_dataset_v2.0/gallery/drink_label_all.txt` 文件中。与原始的 `drink_dataset_v2.0/gallery/drink_label.txt` 标签文件进行对比，可以发现新增了光明和三元系列牛奶的索引图像。

每一行的文本中，第一个字段表示图像的相对路径，第二个字段表示图像对应的标签信息，中间用 `\t` 键分隔开（注意：有些编辑器会将 `tab` 自动转换为 `空格`，这种情况下会导致文件解析报错）。

<a name="建立新的索引库"></a>

#### 2.3.2 建立新的索引库

使用下面的命令构建新的索引库 `index_all`。

**在命令行中构建索引库**
```shell
paddleclas --build_gallery=True --model_name="PP-ShiTuV2" \
-o IndexProcess.image_root=./drink_dataset_v2.0/gallery/ \
-o IndexProcess.index_dir=./drink_dataset_v2.0/index_all \
-o IndexProcess.data_file=./drink_dataset_v2.0/gallery/drink_label_all.txt
```
其中参数`build_gallery(bool)`控制是否使用索引库构建模式，默认为`False`。

最终构建完毕的新的索引库保存在文件夹 `./drink_dataset_v2.0/index_all` 下。具体 `yaml` 请参考[向量检索文档](../deployment/PP-ShiTu/vector_search.md)。

<a name="基于新的索引库的图像识别"></a>

#### 2.3.3 基于新的索引库的图像识别

使用新的索引库，重新对 `mosilian.jpeg` 图像进行识别，运行命令如下。

**在命令行中进行识别和检索**
```shell
paddleclas --model_name=PP-ShiTuV2 --predict_type=shitu \
-o Global.infer_imgs='./drink_dataset_v2.0/test_images/mosilian.jpeg' \
-o IndexProcess.index_dir='./drink_dataset_v2.0/index_all'
```
输出结果如下。

```log
[{'bbox': [290, 297, 564, 919], 'rec_docs': '光明_莫斯利安', 'rec_scores': 0.59137374}], filename: ./drink_dataset_v2.0/test_images/mosilian.jpeg
```


<a name="5"></a>

### 2.4 服务端识别模型列表

目前，我们更推荐您使用[轻量级通用主体检测模型与轻量级通用识别模型](#轻量级通用主体检测模型与轻量级通用识别模型)，以获得更好的测试结果。但是如果您希望体验服务端识别模型，服务器端通用主体检测模型与各方向识别模型、测试数据下载地址以及对应的配置文件地址如下。

| 模型简介         | 推荐场景       | inference 模型   | 预测配置文件                                                             |
| ---------------- | -------------- | ------------  | ----------- |
| 通用主体检测模型 | 通用场景       | [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar)    | -                                                                        |
| Logo 识别模型    | Logo 场景      | [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/logo_rec_ResNet50_Logo3K_v1.0_infer.tar)       | [inference_logo.yaml](../../../deploy/configs/inference_logo.yaml)       |
| 动漫人物识别模型 | 动漫人物场景   | [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/cartoon_rec_ResNet50_iCartoon_v1.0_infer.tar)  | [inference_cartoon.yaml](../../../deploy/configs/inference_cartoon.yaml) |
| 车辆细分类模型   | 车辆场景       | [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/vehicle_cls_ResNet50_CompCars_v1.0_infer.tar)  | [inference_vehicle.yaml](../../../deploy/configs/inference_vehicle.yaml) |
| 商品识别模型     | 商品场景       | [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/product_ResNet50_vd_aliproduct_v1.0_infer.tar) | [inference_product.yaml](../../../deploy/configs/inference_product.yaml) |
| 车辆 ReID 模型   | 车辆 ReID 场景 | [模型下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/vehicle_reid_ResNet50_VERIWild_v1.0_infer.tar) | [inference_vehicle.yaml](../../../deploy/configs/inference_vehicle.yaml) |

可以按照如下命令下载上述模型到 `deploy/models` 文件夹中，以供识别任务使用
```shell
cd ./deploy
mkdir -p models

cd ./models
# 下载服务器端通用主体检测模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar && tar -xf ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
# 下载通用识别模型并解压
wget {识别模型下载链接地址} && tar -xf {压缩包的名称}
```

然后使用如下命令下载各个识别场景的测试数据：

```shell
# 回到 deploy 目录下
cd ..
# 下载测试数据并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/recognition_demo_data_en_v1.1.tar && tar -xf recognition_demo_data_en_v1.1.tar
```

解压完毕后，`recognition_demo_data_v1.1` 文件夹下应有如下文件结构：

```log
├── recognition_demo_data_v1.1
│   ├── gallery_cartoon
│   ├── gallery_logo
│   ├── gallery_product
│   ├── gallery_vehicle
│   ├── test_cartoon
│   ├── test_logo
│   ├── test_product
│   └── test_vehicle
├── ...
```

按照上述步骤下载模型和测试数据后，您可以重新建立索引库，并进行相关方向识别模型的测试。

* 更多关于主体检测的介绍可以参考：[主体检测教程文档](../training/PP-ShiTu/mainbody_detection.md)；关于特征提取的介绍可以参考：[特征提取教程文档](../training/PP-ShiTu/feature_extraction.md)；关于向量检索的介绍可以参考：[向量检索教程文档](../deployment/PP-ShiTu/vector_search.md)。
