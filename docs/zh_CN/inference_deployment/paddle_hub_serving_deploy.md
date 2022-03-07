# 基于 PaddleHub Serving 的服务部署

PaddleClas 支持通过 PaddleHub 快速进行服务化部署。目前支持图像分类的部署，图像识别的部署敬请期待。

---


## 目录
- [1. 简介](#1)
- [2. 准备环境](#2)
- [3. 下载推理模型](#3)
- [4. 安装服务模块](#4)
- [5. 启动服务](#5)
    - [5.1 命令行命令启动](#5.1)
    - [5.2 配置文件启动](#5.2)
- [6. 发送预测请求](#6)
- [7. 自定义修改服务模块](#7)


<a name="1"></a>
## 1. 简介

hubserving 服务部署配置服务包 `clas` 下包含 3 个必选文件，目录如下：

```
hubserving/clas/
  └─  __init__.py    空文件，必选
  └─  config.json    配置文件，可选，使用配置启动服务时作为参数传入
  └─  module.py      主模块，必选，包含服务的完整逻辑
  └─  params.py      参数文件，必选，包含模型路径、前后处理参数等参数
```


<a name="2"></a>
## 2. 准备环境
```shell
# 安装 paddlehub,请安装 2.0 版本
pip3 install paddlehub==2.1.0 --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```


<a name="3"></a>
## 3. 下载推理模型

安装服务模块前，需要准备推理模型并放到正确路径，默认模型路径为：

* 分类推理模型结构文件：`PaddleClas/inference/inference.pdmodel`
* 分类推理模型权重文件：`PaddleClas/inference/inference.pdiparams`

**注意**：
* 模型文件路径可在 `PaddleClas/deploy/hubserving/clas/params.py` 中查看和修改：

  ```python
  "inference_model_dir": "../inference/"
  ```
需要注意，
  * 模型文件（包括 `.pdmodel` 与 `.pdiparams`）名称必须为 `inference`。
  * 我们也提供了大量基于 ImageNet-1k 数据集的预训练模型，模型列表及下载地址详见[模型库概览](../algorithm_introduction/ImageNet_models.md)，也可以使用自己训练转换好的模型。


<a name="4"></a>
## 4. 安装服务模块

针对 Linux 环境和 Windows 环境，安装命令如下。

* 在 Linux 环境下，安装示例如下：
```shell
cd PaddleClas/deploy
# 安装服务模块：
hub install hubserving/clas/
```

* 在 Windows 环境下(文件夹的分隔符为`\`)，安装示例如下：

```shell
cd PaddleClas\deploy
# 安装服务模块：  
hub install hubserving\clas\
```


<a name="5"></a>
## 5. 启动服务


<a name="5.1"></a>
### 5.1 命令行命令启动

该方式仅支持使用 CPU 预测。启动命令：

```shell
$ hub serving start --modules Module1==Version1 \
                    --port XXXX \
                    --use_multiprocess \
                    --workers \
```  

**参数说明**：
|参数|用途|  
|-|-|  
|--modules/-m| [**必选**] PaddleHub Serving 预安装模型，以多个 Module==Version 键值对的形式列出<br>*`当不指定 Version 时，默认选择最新版本`*|  
|--port/-p| [**可选**] 服务端口，默认为 8866|  
|--use_multiprocess| [**可选**] 是否启用并发方式，默认为单进程方式，推荐多核 CPU 机器使用此方式<br>*`Windows 操作系统只支持单进程方式`*|
|--workers| [**可选**] 在并发方式下指定的并发任务数，默认为 `2*cpu_count-1`，其中 `cpu_count` 为 CPU 核数|  

如按默认参数启动服务：```hub serving start -m clas_system```  

这样就完成了一个服务化 API 的部署，使用默认端口号 8866。


<a name="5.2"></a>
### 5.2 配置文件启动

该方式仅支持使用 CPU 或 GPU 预测。启动命令：

```hub serving start -c config.json```  

其中，`config.json` 格式如下：

```json
{
    "modules_info": {
        "clas_system": {
            "init_args": {
                "version": "1.0.0",
                "use_gpu": true,
                "enable_mkldnn": false
            },
            "predict_args": {
            }
        }
    },
    "port": 8866,
    "use_multiprocess": false,
    "workers": 2
}
```

**参数说明**：
* `init_args` 中的可配参数与 `module.py` 中的 `_initialize` 函数接口一致。其中，
  - 当 `use_gpu` 为 `true` 时，表示使用 GPU 启动服务。
  - 当 `enable_mkldnn` 为 `true` 时，表示使用 MKL-DNN 加速。
* `predict_args` 中的可配参数与 `module.py` 中的 `predict` 函数接口一致。

**注意**：
* 使用配置文件启动服务时，将使用配置文件中的参数设置，其他命令行参数将被忽略；
* 如果使用 GPU 预测(即，`use_gpu` 置为 `true`)，则需要在启动服务之前，设置 `CUDA_VISIBLE_DEVICES` 环境变量来指定所使用的 GPU 卡号，如：`export CUDA_VISIBLE_DEVICES=0`；
* **`use_gpu` 不可与 `use_multiprocess` 同时为 `true`**；
* **`use_gpu` 与 `enable_mkldnn` 同时为 `true` 时，将忽略 `enable_mkldnn`，而使用 GPU**。

如使用 GPU 3 号卡启动服务：

```shell
cd PaddleClas/deploy
export CUDA_VISIBLE_DEVICES=3
hub serving start -c hubserving/clas/config.json
```

<a name="6"></a>
## 6. 发送预测请求

配置好服务端后，可使用以下命令发送预测请求，获取预测结果：

```shell
cd PaddleClas/deploy
python hubserving/test_hubserving.py server_url image_path
```  

**脚本参数说明**：
* **server_url**：服务地址，格式为  
`http://[ip_address]:[port]/predict/[module_name]`  
* **image_path**：测试图像路径，可以是单张图片路径，也可以是图像集合目录路径。
* **batch_size**：[**可选**] 以 `batch_size` 大小为单位进行预测，默认为 `1`。
* **resize_short**：[**可选**] 预处理时，按短边调整大小，默认为 `256`。
* **crop_size**：[**可选**] 预处理时，居中裁剪的大小，默认为 `224`。
* **normalize**：[**可选**] 预处理时，是否进行 `normalize`，默认为 `True`。
* **to_chw**：[**可选**] 预处理时，是否调整为 `CHW` 顺序，默认为 `True`。

**注意**：如果使用 `Transformer` 系列模型，如 `DeiT_***_384`, `ViT_***_384` 等，请注意模型的输入数据尺寸，需要指定`--resize_short=384 --crop_size=384`。

访问示例：

```shell
python hubserving/test_hubserving.py --server_url http://127.0.0.1:8866/predict/clas_system --image_file ./hubserving/ILSVRC2012_val_00006666.JPEG --batch_size 8
```

**返回结果格式说明**：
返回结果为列表（list），包含 top-k 个分类结果，以及对应的得分，还有此图片预测耗时，具体如下：
```
list: 返回结果
└─ list: 第一张图片结果
   └─ list: 前 k 个分类结果，依 score 递减排序
   └─ list: 前 k 个分类结果对应的 score，依 score 递减排序
   └─ float: 该图分类耗时，单位秒
```


<a name="7"></a>
## 7. 自定义修改服务模块

如果需要修改服务逻辑，需要进行以下操作：  

1. 停止服务  
```hub serving stop --port/-p XXXX```  

2. 到相应的 `module.py` 和 `params.py` 等文件中根据实际需求修改代码。`module.py` 修改后需要重新安装（`hub install hubserving/clas/`）并部署。在进行部署前，可通过 `python hubserving/clas/module.py` 测试已安装服务模块。

3. 卸载旧服务包  
```hub uninstall clas_system```  

4. 安装修改后的新服务包  
```hub install hubserving/clas/```  

5.重新启动服务  
```hub serving start -m clas_system```  

**注意**：
常用参数可在 `PaddleClas/deploy/hubserving/clas/params.py` 中修改：
  * 更换模型，需要修改模型文件路径参数:
    ```python
    "inference_model_dir":
    ```
  * 更改后处理时返回的 `top-k` 结果数量：
    ```python
    'topk':
    ```
  * 更改后处理时的 lable 与 class id 对应映射文件：
    ```python
    'class_id_map_file':
    ```

为了避免不必要的延时以及能够以 batch_size 进行预测，数据预处理逻辑（包括 `resize`、`crop` 等操作）均在客户端完成，因此需要在 `PaddleClas/deploy/hubserving/test_hubserving.py#L35-L52` 中修改。
