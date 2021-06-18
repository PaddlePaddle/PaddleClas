
# 基于Python预测引擎推理

inference 模型（`paddle.jit.save`保存的模型）
一般是模型训练，把模型结构和模型参数保存在文件中的固化模型，多用于预测部署场景。
训练过程中保存的模型是checkpoints模型，保存的只有模型的参数，多用于恢复训练等。
与checkpoints模型相比，inference 模型会额外保存模型的结构信息，在预测部署、加速推理上性能优越，灵活方便，适合于实际系统集成。

接下来首先介绍如何将训练的模型转换成inference模型，然后将依次介绍主体检测、特征提取在CPU、GPU上的预测方法，
之后介绍了主体检测、特征提取、特征检索串联的预测方法，最后介绍了图像分类的预测方法。


- [一、训练模型转inference模型](#训练模型转inference模型)
    - [1. 特征提取模型转inference模型](#特征提取模型转inference模型)  
    - [2. 分类模型转inference模型](#分类模型转inference模型)

- [二、主体检测模型推理](#主体检测模型推理)

- [三、特征提取模型推理](#特征提取模型推理)

- [四、主体检测、特征提取和向量检索串联](#主体检测、特征提取和向量检索串联)

- [五、图像分类模型推理](#图像分类模型推理)


<a name="训练模型转inference模型"></a>
## 一、训练模型转inference模型

<a name="特征提取模型转inference模型"></a>
### 特征提取模型转inference模型
以下命令请在PaddleClas的根目录执行。以商品识别特征提取模型模型为例，首先下载预训练模型：

```shell script
wget -P ./product_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained.pdparams
```

上述模型是ResNet50_vd在AliProduct上训练的模型，训练使用的配置文件为ppcls/configs/Products/ResNet50_vd_Aliproduct.yaml
将训练好的模型转换成inference模型只需要运行如下命令：
``` shell script
# -c 后面设置训练算法的yml配置文件
# -o 配置可选参数
# Global.pretrained_model 参数设置待转换的训练模型地址，不用添加文件后缀 .pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3.7 tools/export_model.py -c ppcls/configs/Products/ResNet50_vd_Aliproduct.yaml -o Global.pretrained_model=./product_pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained Global.save_inference_dir=./deploy/models/product_ResNet50_vd_aliproduct_v1.0_infer
```

这里也可以使用自己训练的模型。转inference模型时，使用的配置文件和训练时使用的配置文件相同。另外，还需要设置配置文件中的`Global.pretrained_model`参数，其指向训练中保存的模型参数文件。
转换成功后，在模型保存目录下有三个文件：
``` 
├── product_ResNet50_vd_aliproduct_v1.0_infer
│   ├── inference.pdiparams         # 识别inference模型的参数文件
│   ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
│   └── inference.pdmodel           # 识别inference模型的program文件
```

<a name="分类模型转inference模型"></a>
### 分类模型转inference模型

下载预训练模型：
``` shell script
wget -P ./cls_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
```

上述模型是使用ResNet50_vd在ImageNet上训练的模型，使用的配置文件为`ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml`。
转inference模型与特征提取模型的方式相同，如下：
```
# -c 后面设置训练算法的yml配置文件
# -o 配置可选参数
# Global.pretrained_model 参数设置待转换的训练模型地址，不用添加文件后缀 .pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3.7 tools/export_model.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml -o Global.pretrained_model=./cls_pretrain/ResNet50_vd_pretrained  Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer
```

**注意：**如果您是在自己的数据集上训练的模型，并且调整了中文字符的字典文件，请注意修改配置文件中的`character_dict_path`是否是所需要的字典文件。

转换成功后，在目录下有三个文件：
```
├── class_ResNet50_vd_ImageNet_infer
│   ├── inference.pdiparams         # 识别inference模型的参数文件
│   ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
│   └── inference.pdmodel           # 识别inference模型的program文件
```

<a name="主体检测模型推理"></a>
## 二、主体检测模型推理

下面介绍主体检测模型推理，以下命令请进入PaddleClas的deploy目录执行：
```shell script
cd deploy
```
使用PaddleClas提供的主体检测Inference模型进行推理，可以执行：

```shell script
mkdir -p models
cd models
# 下载通用检测inference模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar && tar -xf ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
cd ..

# 用下载的inference模型进行预测
python3.7 python/predict_det.py -c configs/inference_det.yaml
```
输入的图像如下所示。
[](../images/recognition/product_demo/wangzai.jpg)
最终输出结果如下：
```text
[{'class_id': 0, 'score': 0.4762245, 'bbox': array([305.55115, 226.05322, 776.61084, 930.42395], 
dtype=float32), 'label_name': 'foreground'}]
```
检测的可视化结果如下：
[](../images/recognition/product_demo/wangzai_det_result.jpg)

如果想要修改图像，可以在configs/inference_det.yaml中，修改infer_imgs的值，或使用-o Global.infer_imgs修改，
例如，要使用`images/anmuxi.jpg`可以运行：

```shell script
python3.7 python/predict_det.py -c configs/inference_det.yaml -o Global.infer_imgs=images/anmuxi.jpg
```

如果想使用CPU进行预测，可以将配置文件中use_gpu选项设置为False，或者执行命令：
```
python3 tools/infer/predict_det.py  -o Global.use_gpu=False
```

## 三、特征提取模型推理

下面以商品特征提取为例，介绍特征提取模型推理。其他应用可以参考图像识别快速开始中的[模型地址](./tutorials/quick_start_recognition.md#2-%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB%E4%BD%93%E9%AA%8C)，
将链接替换为相应模型的链接。以下命令请进入PaddleClas的deploy目录执行：
```shell script
cd deploy
```
使用PaddleClas提供的商品特征提取Inference模型进行推理，可以执行：

```shell script
mkdir -p models
cd models
# 下载商品特征提取inference模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/product_ResNet50_vd_aliproduct_v1.0_infer.tar && tar -xf product_ResNet50_vd_aliproduct_v1.0_infer.tar
cd ..

# 用下载的inference模型进行预测
python3.7 python/predict_rec.py -c configs/inference_rec.yaml
```

如果想要修改图像，可以在configs/inference_det.yaml中，修改infer_imgs的值，或使用-o Global.infer_imgs修改，
例如，要使用`images/anmuxi.jpg`可以运行：

```shell script
python3.7 python/predict_rec.py -c configs/inference_rec.yaml -o Global.infer_imgs=images/anmuxi.jpg
```

如果想使用CPU进行预测，可以将配置文件中use_gpu选项设置为False，或者执行命令：
```
python3 tools/infer/predict_rec.py  -o Global.use_gpu=False
```

<a name="主体检测、特征提取和向量检索串联"></a>
## 四、主体检测、特征提取和向量检索串联
主体检测、特征提取和向量检索的串联预测，可以参考[图像识别快速体验](./tutorials/quick_start_recognition.md)

<a name="图像分类模型推理"></a>
## 五、图像分类模型推理

下面介绍图像分类模型推理，以下命令请进入PaddleClas的deploy目录执行：
```shell script
cd deploy
```
使用PaddleClas提供的商品特征提取Inference模型进行推理，首先请下载预训练模型并导出inference模型，具体参见[2. 分类模型转inference模型](#分类模型转inference模型)。

导出inference模型后，可以使用下面的命令预测：
```shell script

python3.7 python/predict_cls.py -c configs/inference_rec.yaml
```

如果想要修改图像，可以在configs/inference_det.yaml中，修改infer_imgs的值，或使用-o Global.infer_imgs修改，
例如，要使用`images/ILSVRC2012_val_00010010.jpeg`可以运行：

```shell script
python3.7 python/predict_cls.py -c configs/inference_rec.yaml -o Global.infer_imgs=images/ILSVRC2012_val_00010010.jpeg

```

如果想使用CPU进行预测，可以将配置文件中use_gpu选项设置为False，或者执行命令：
```
python3 tools/infer/predict_rec.py  -o Global.use_gpu=False
```

