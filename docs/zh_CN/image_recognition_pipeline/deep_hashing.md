# 哈希编码

最近邻搜索是指在数据库中查找与查询数据距离最近的点，在计算机视觉、推荐系统、机器学习等领域中广泛使用。在`PP-ShiTu`中，输入图像经过主体检测模型去掉背景后，再经过特征提取模型提取特征，之后经过检索得到输入图像的类别。在这个过程中，一般来说，提取的特征是`float32`数据类型。当离线特征库中存储的`feature`比较多时，就占用较大的存储空间，同时检索过程也会变慢。如果利用`哈希编码`将特征由`float32`转成`0`或者`1`表示的二值特征，那么不仅降低存储空间，同时也能大大加快检索速度。

哈希编码，主要用在`PP-ShiTu`的**特征提取模型**部分，将模型输出特征直接二值化。即训练特征提取模型时，将模型的输出映射到二值空间。

注意，由于使用二值特征表示图像特征，精度可能会下降，请根据实际情况，酌情使用。


## 目录

- [1. 特征模型二值特征训练](#1)
	- [1.1 PP-ShiTu特征提取模型二值训练](#1.1)
	- [1.2 其他特征模型二值训练](#1.2)
- [2. 检索算法配置](#2)

<a name="1"></a>

## 1. 特征模型二值特征训练

<a name="1.1"></a>

注意，此模块目前只支持`PP-ShiTuV1`,`PP-ShiTuV2`暂未适配。

### 1.1 PP-ShiTu特征提取模型二值训练

PP-ShiTu特征提取模型二值特征模型，配置文件位于`ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_binary.yaml`，相关训练方法如下。

```shell
# 单卡 GPU
python3.7 tools/train.py \
-c ./ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_binary.yaml \
-o Arch.Backbone.pretrained=True \
-o Global.device=gpu

# 多卡 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch tools/train.py \
-c ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_binary.yaml \
-o Arch.Backbone.pretrained=True \
-o Global.device=gpu
```

其中`数据准备`、`模型评估`等，请参考[此文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models_training/recognition.md)。

<a name="1.2"></a>

### 1.2 其他特征模型二值训练

其他二值特征训练模型的配置文件位于`ppcls/configs/DeepHash/`文件夹下，此文件夹下的相关配置文件主要是复现相关`deep hashing`相关算法。包括：`DCH, DSHSD, LCDSH`三种算法。这三种算法相关介绍，详见[Deep Hashing相关算法介绍](../algorithm_introduction/deep_hashing_introduction.md)。

相关训练方法，请参考[分类模型训练文档](../models_training/classification.md)。

<a name="2"></a>

## 2. 检索算法配置

在PP-ShiTu中使用二值特征，部署及离线推理配置请参考`deploy/configs/inference_general_binary.yaml`。配置文件中相关参数介绍请参考[向量检索文档](./vector_search.md).

其中需值得注意的是，二值检索相关配置应设置如下：

```yaml
IndexProcess:
  index_method: "FLAT" # supported: HNSW32, IVF, Flat
  delimiter: "\t"
  dist_type: "hamming"
  hamming_radius: 100
```

其中`hamming_radius`可以根据自己实际精度要求，适当调节。
