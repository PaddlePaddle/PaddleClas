# 向量检索

向量检索是在图像识别、图像检索中应用比较广泛。其主要目标是，对于给定的查询向量，在已经建立好的向量库中，与库中所有的待查询向量，进行特征向量的相似度或距离计算，得到相似度排序。在图像识别系统中，我们使用[Faiss](https://github.com/facebookresearch/faiss)对此部分进行支持，具体信息请详查[Faiss官网](https://github.com/facebookresearch/faiss)。`Faiss`主要有以下优势

- 适配性好：支持Windos、Linux、MacOS系统
- 安装方便： 支持`python`接口，直接使用`pip`安装
- 算法丰富：支持多种检索算法，满足不同场景的需求
- 同时支持CPU、GPU，能够加速检索过程

值得注意的是，为了更好是适配性，目前版本，`PaddleClas`中暂时**只使用CPU进行向量检索**。

本文档主要主要介绍PaddleClas中检索模块的安装、使用的检索算法，及使用过程中的相关配置文件中参数介绍。

## 一、检索库安装

`Faiss`具体安装方法如下：

```python
pip install faiss-cpu==1.7.1post2
```

若使用时，不能正常引用，则`uninstall` 之后，重新`install`，尤其是`windows`下。

## 二、使用的检索算法

目前`PaddleClas`中检索模块，支持如下三种检索算法

- **HNSW32**: 一种图索引方法。检索精度较高，速度较快。但是特征库只支持添加图像功能，不支持删除图像特征功能。（默认方法）
- **IVF**：倒排索引检索方法。速度较快，但是精度略低。特征库支持增加、删除图像特功能。
- **FLAT**： 暴力检索算法。精度最高，但是数据量大时，检索速度较慢。特征库支持增加、删除图像特征功能。

每种检索算法，满足不同场景。其中`HNSW32`为默认方法，此方法在检索精度、检索速度可以取得一个较好的平衡，具体算法介绍可以查看[官方文档](https://github.com/facebookresearch/faiss/wiki)。

## 三、相关配置文档参数介绍

涉及检索模块配置文件位于：`deploy/configs/`下，其中`build_*.yaml`是建立特征库的相关配置文件，`inference_*.yaml`是检索或者分类的推理配置文件。

### 3.1 建库配置文件参数

示例建库的配置如下

```yaml
# indexing engine config
IndexProcess:
  index_method: "HNSW32" # supported: HNSW32, IVF, Flat
  index_dir: "./recognition_demo_data_v1.1/gallery_product/index"
  image_root: "./recognition_demo_data_v1.1/gallery_product/"
  data_file:  "./recognition_demo_data_v1.1/gallery_product/data_file.txt"
  index_operation: "new" # suported: "append", "remove", "new"
  delimiter: "\t"
  dist_type: "IP"
  embedding_size: 512
```

- **index_method**：使用的检索算法。目前支持三种，HNSW32、IVF、Flat
- **index_dir**：构建的特征库所存放的文件夹
- **image_root**：构建特征库所需要的标注图像所存储的文件夹位置
- **data_file**：构建特征库所需要的标注图像的数据列表，每一行的格式：relative_path label
- **index_operation**： 此次运行建库的操作：`new`新建，`append`将data_file的图像特征添加到特征库中，`remove`将data_file的特征从特征库中删除
- **delimiter**：**data_file**中每一行的间隔符
- **dist_type**: 特征屁配过程中使用的相似度计算方式。`IP`内积相似度计算方式，`L2`欧式距离计算方法
- **embedding_size**：特征维度

### 3.2 检索配置文件参数

```yaml
IndexProcess:
  index_dir: "./recognition_demo_data_v1.1/gallery_logo/index/"
  return_k: 5
  score_thres: 0.5
```

与建库配置文件相似，新参数主要如下：

- `return_k`: 检索结果返回`k`个结果
- `score_thres`: 检索匹配的阈值
