# PP-ShiTu 库管理工具

本工具是PP-ShiTu的离线库管理工具，主要功能包括：新建图像库、更改图像库、建立索引库、更新索引库等功能。此工具是为了用户能够可视化的管理图像及对应的index库，用户可根据实际情况，灵活的增删改查相应的gallery图像库及索引文件，在提升用户体验的同时，辅助PP-ShiTu在实际应用的过程中达到更好的效果。

目前此工具支持平台包括：

- Mac
- Windows
- Linux（注意，由于linux输入法问题，可能无法支持中文）

## 目录

- [1. 功能介绍](#1)

  - [1.1 新建图像库](#1.1)
  - [1.2 打开图像库](#1.2)
  - [1.3 导入图像](#1.3)
  - [1.4 图像操作](#1.3)

  - [1.5 其他功能](#1.5)

- [2. 使用说明](#2)

  - [2.1 环境安装](#2.1)
  - [2.2 模型及数据准备](#2.2)
  - [2.3运行使用](#2.3)

- [3.生成文件介绍](#3)

- [致谢](#4)

- [FAQ](#FAQ)

<a name="1"></a>

## 1. 功能介绍

此工具主要功能包括：

- 构建`PP-ShiTu`中索引库对应的`gallery`图像库
- 根据构建的`gallery`图像库，生成索引库
- 对`gallery`图像库进行操作，如增删改查等操作，并更新对应的索引库

其中主界面的按钮如下图所示

<div align="center">
<img src="https://user-images.githubusercontent.com/11568925/188273082-b1ada7ed-e56e-4b6a-9e79-2dda01a3db69.png"  width = "600" />
<p>界面按钮展示</p>
</div>

上图中第一行包括：`主要功能按钮`、`保存按钮`、`新增类别按钮`、`删减类别按钮`。

第二行包括：`搜索框`、`搜索确定键`、`新加图像按钮`、`删除图像按钮`。

下面将进行具体功能介绍，其操作入口，可以点击`主要功能按钮`下拉菜单查看，如下图所示：

<div align="center">
<img src="https://user-images.githubusercontent.com/11568925/188273056-04b376f5-7275-47ac-898b-474a667bc6a7.png"  width = "600" />
<p>主要功能展示</p>
</div>

<a name="1.1"></a>

### 1.1 新建图像库

点击新建库功能后，会选择一个**空的存储目录**或者**新建目录**，此时所有的图片及对应的索引库都会存放在此目录下。完成操作后，如下图所示

<div align="center">
<img src="https://user-images.githubusercontent.com/11568925/188273108-8789b8cf-d2ab-49d5-bc82-f0bf7b41c686.png"  width = "600" />
<p>新建库</p>
</div>

此时，用户可以新建类别具体可以点击`新增类别按钮`、`删减类别按钮`。选中类别后，可以进行添加图像及相关操作，具体可以点击及`新加图像按钮`、`删除图像按钮`。完成操作后，**注意保存**。

<a name="1.2"></a>

### 1.2 打开图像库

此功能是，用此工具存储好的库，进行打开编辑。注意，**打开库时，请选择打开的是新建库时文件夹路径**。打开库后，示例如下

<div align="center">
<img src="https://user-images.githubusercontent.com/11568925/188273143-00ff0558-ccc9-4b8d-9364-43eef5dce334.png"  width = "600" />
<p>打开库</p>
</div>

<a name="1.3"></a>

### 1.3 导入图像

在打开图像库或者新建图像库完成后，可以使用导入图像功能，即导入用户自己生成好的图像库。具体有支持两种导入格式

- image_list格式：打开具体的`.txt`文件。`.txt`文件中每一行格式： `image_path label`。跟据文件路径及label导入
- 多文件夹格式：打开`具体文件夹`，此文件夹下存储多个子文件夹，每个子文件夹名字为`label_name`，每个子文件夹中存储对应的图像数据。

<a name="1.4"></a>

### 1.4 图像操作

选择图像后，鼠标右击可以进行如下操作，可以根据需求，选择具体的操作，**注意修改完成图像后，请点击保存按钮，进行保存**

<div align="center">
<img src="https://user-images.githubusercontent.com/11568925/188273178-5eff2f2e-7a8b-4a2b-809e-78f99479162d.png"  width = "600" />
<p>图像操作</p>
</div>

<a name="1.5"></a>

### 1.5 生成、更新index库

在用户完成图像库的新建、打开或者修改，并完成保存操作后。可以点击`主要功能按钮`中`新建/重建索引库`、`更新索引库`等功能，进行索引库的新建或者更新，生成`PP-ShiTu`使用的Index库

<a name="2"></a>

## 2. 使用说明

<a name="2.1"></a>

### 2.1 环境安装

安装好`PaddleClas`后

```shell
pip install fastapi
pip install uvicorn
pip install pyqt5
pip install psutil
```

<a name="2.2"></a>

### 2.2 模型及数据准备

请按照[PP-ShiTu快速体验](../../quick_start/quick_start_recognition.md#2.2.1)中下载及准备inference model，并修改好`${PaddleClas}/deploy/configs/inference_drink.yaml`的相关参数，同时准备好数据集。在具体使用时，请替换好自己的数据集及模型文件。

```shell
cd ${PaddleClas}/deploy/shitu_index_manager
mkdir models
cd models
# 下载及解压识别模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0_infer.tar && tar -xf general_PPLCNetV2_base_pretrained_v1.0_infer.tar
cd ..
# 下载及解压示例数据集
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v2.0.tar && tar -xf drink_dataset_v2.0.tar
```

<a name="2.3"></a>

### 2.3 运行使用

运行方式如下

```shell
cd ${PaddleClas}/deploy/shitu_index_manager
cp ../configs/inference_drink.yaml .
# 注意如果没有按照2.2中准备数据集及代码，请手动修改inference_drink.yaml，做好适配
python index_manager.py -c inference_drink.yaml
```

运行成功后，会自动跳转到工具界面，可以按照如下步骤，生成新的index库。

1. 点击菜单栏`新建图像库`，会提示打开一个文件夹，此时请创建一个**新的文件夹**，并打开。如在`${PaddleClas}/deploy/shitu_index_manager`下新建一个`drink_index`文件夹
2. 导入图像，或者如上面功能介绍，自己手动新增类别和相应的图像，下面介绍两种导入图像方式，操作时，二选一即可。
   - 点击`导入图像`->`导入image_list图像`，打开`${PaddleClas}/deploy/shitu_index_manager/drink_dataset_v2.0/gallery/drink_label.txt`，此时就可以将`drink_label.txt`中的图像全部导入进来，图像类别就是`drink_label.txt`中记录的类别。
   - 点击`导入图像`->`导入多文件夹图像`，打开`${PaddleClas}/deploy/shitu_index_manager/drink_dataset_v2.0/gallery/`文件夹，此时就将`gallery`文件夹下，所有子文件夹都导入进来，图像类别就是子文件夹的名字。
3. 点击菜单栏中`新建/重建 索引库`，此时就会开始生成索引库。如果图片较多或者使用cpu来进行特征提取，那么耗时会比较长，请耐心等待。
4. 生成索引库成功后，会发现在`drink_index`文件夹下生成如[3](#3) 中介绍的文件，此时`index`子文件夹下生出的文件，就是`PP-ShiTu`所使用的索引文件。

**注意**：

- 利用此工具生成的index库，如`drink_index`文件夹，请妥善存储。之后，可以继续使用此工具中`打开图像库`功能，打开`drink_index`文件夹，继续对index库进行增删改查操作，具体功能可以查看[功能介绍](#1)。
- 打开一个生成好的库，在其上面进行增删改查操作后，请及时保存。保存后并及时使用菜单中`更新索引库`功能，对索引库进行更新
- 如果要使用自己的图像库文件，图像生成格式如示例数据格式，生成`image_list.txt`或者多文件夹存储，二选一。

<a name="3"></a>

## 3. 生成文件介绍

使用此工具后，会生成如下格式的文件

```shell
index_root/            # 库存储目录
|-- image_list.txt     # 图像列表，每行：image_path label。由前端生成及修改，后端只读
|-- images             # 图像存储目录，由前端生成及增删查等操作。后端只读
|   |-- md5.jpg
|   |-- md5.jpg  
|   |-- ……  
|-- features.pkl       # 建库之后，保存的embedding向量，后端生成，前端无需操作
|-- index              # 真正的生成的index库存储目录，后端生成及操作，前端无需操作。
|   |-- vector.index   # faiss生成的索引库
|   |-- id_map.pkl     # 索引文件
```

其中`index_root`是使用此工具时，用户选择的存储目录，库的索引文件存储在`index`文件夹中。

使用`PP-ShiTu`时，索引文件目录需换成`index`文件夹的地址。

<a name="4"></a>

## 致谢

此工具的前端主要由[国内qt论坛](http://www.qtcn.org/)总版主[小熊宝宝](https://github.com/cnhemiya)完成，感谢**小熊宝宝**的大力支持~~

此工具前端原项目地址：https://github.com/cnhemiya/shitu-manager

<a name="FAQ"></a>

## FAQ

- 问题1: 点击新建索引库后，程序假死

  答：生成索引库比较耗时，耐心等待一段时间就好

- 问题2: 导入图像是什么格式？

  答： 目前支持两种格式 1）image_list 格式，list中每行格式：path label。2）文件夹格式：类似`ImageNet`存储方式

- 问题3: 生成 index库报错

  答：在修改图像后，必须点击保存按钮，保存完成后，再继续生成index库。

- 问题4: 报错 图像与index库不一致

  答：可能用户自己修改了image_list.txt，修改完成后，请及时更新index库，保证其一致。
