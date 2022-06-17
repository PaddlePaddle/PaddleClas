简体中文 | [English](../../en/inference_deployment/classification_serving_deploy_en.md)

# 分类模型服务化部署

## 目录

- [1. 简介](#1-简介)
- [2. Serving 安装](#2-serving-安装)
- [3. 图像分类服务部署](#3-图像分类服务部署)
- [3.1 模型转换](#31-模型转换)
- [3.2 服务部署和请求](#32-服务部署和请求)
    - [3.2.1 Python Serving](#321-python-serving)
    - [3.2.2 C++ Serving](#322-c-serving)
- [4.FAQ](#4faq)

<a name="1"></a>
## 1. 简介

[Paddle Serving](https://github.com/PaddlePaddle/Serving) 旨在帮助深度学习开发者轻松部署在线预测服务，支持一键部署工业级的服务能力、客户端和服务端之间高并发和高效通信、并支持多种编程语言开发客户端。

该部分以 HTTP 预测服务部署为例，介绍怎样在 PaddleClas 中使用 PaddleServing 部署模型服务。目前只支持 Linux 平台部署，暂不支持 Windows 平台。

<a name="2"></a>
## 2. Serving 安装

Serving 官网推荐使用 docker 安装并部署 Serving 环境。首先需要拉取 docker 环境并创建基于 Serving 的 docker。

```shell
# 启动GPU docker
docker pull paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel
nvidia-docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel bash
nvidia-docker exec -it test bash

# 启动CPU docker
docker pull paddlepaddle/serving:0.7.0-devel
docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-devel bash
docker exec -it test bash
```

进入 docker 后，需要安装 Serving 相关的 python 包。
```shell
python3.7 -m pip install paddle-serving-client==0.7.0
python3.7 -m pip install paddle-serving-app==0.7.0
python3.7 -m pip install faiss-cpu==1.7.1post2

#若为CPU部署环境:
python3.7 -m pip install paddle-serving-server==0.7.0 # CPU
python3.7 -m pip install paddlepaddle==2.2.0          # CPU

#若为GPU部署环境
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post102 # GPU with CUDA10.2 + TensorRT6
python3.7 -m pip install paddlepaddle-gpu==2.2.0     # GPU with CUDA10.2

#其他GPU环境需要确认环境再选择执行哪一条
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post101 # GPU with CUDA10.1 + TensorRT6
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post112 # GPU with CUDA11.2 + TensorRT8
```

* 如果安装速度太慢，可以通过 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 更换源，加速安装过程。
* 其他环境配置安装请参考：[使用Docker安装Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Install_CN.md)

<a name="3"></a>

## 3. 图像分类服务部署

下面以经典的 ResNet50_vd 模型为例，介绍如何部署图像分类服务。

<a name="3.1"></a>
### 3.1 模型转换

使用 PaddleServing 做服务化部署时，需要将保存的 inference 模型转换为 Serving 模型。
- 进入工作目录：
  ```shell
  cd deploy/paddleserving
  ```
- 下载并解压 ResNet50_vd 的 inference 模型：
  ```shell
  # 下载 ResNet50_vd inference 模型
  wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar
  # 解压 ResNet50_vd inference 模型
  tar xf ResNet50_vd_infer.tar
  ```
- 用 paddle_serving_client 命令把下载的 inference 模型转换成易于 Server 部署的模型格式：
  ```shell
  # 转换 ResNet50_vd 模型
  python3.7 -m paddle_serving_client.convert \
  --dirname ./ResNet50_vd_infer/ \
  --model_filename inference.pdmodel  \
  --params_filename inference.pdiparams \
  --serving_server ./ResNet50_vd_serving/ \
  --serving_client ./ResNet50_vd_client/
  ```
  上述命令中参数具体含义如下表所示
    | 参数              | 类型 | 默认值             | 描述                                                         |
  | ----------------- | ---- | ------------------ | ------------------------------------------------------------ |
  | `dirname`         | str  | -                  | 需要转换的模型文件存储路径，Program结构文件和参数文件均保存在此目录。 |
  | `model_filename`  | str  | None               | 存储需要转换的模型Inference Program结构的文件名称。如果设置为None，则使用 `__model__` 作为默认的文件名 |
  | `params_filename` | str  | None               | 存储需要转换的模型所有参数的文件名称。当且仅当所有模型参数被保>存在一个单独的二进制文件中，它才需要被指定。如果模型参数是存储在各自分离的文件中，设置它的值为None |
  | `serving_server`  | str  | `"serving_server"` | 转换后的模型文件和配置文件的存储路径。默认值为serving_server |
  | `serving_client`  | str  | `"serving_client"` | 转换后的客户端配置文件存储路径。默认值为serving_client       |

    ResNet50_vd 推理模型转换完成后，会在当前文件夹多出 `ResNet50_vd_serving` 和 `ResNet50_vd_client` 的文件夹，具备如下结构：
  ```shell
  ├── ResNet50_vd_serving/
  │   ├── inference.pdiparams
  │   ├── inference.pdmodel
  │   ├── serving_server_conf.prototxt
  │   └── serving_server_conf.stream.prototxt
  │
  └── ResNet50_vd_client/
      ├── serving_client_conf.prototxt
      └── serving_client_conf.stream.prototxt
  ```

- Serving 为了兼容不同模型的部署，提供了输入输出重命名的功能。让不同的模型在推理部署时，只需要修改配置文件的 `alias_name` 即可，无需修改代码即可完成推理部署。因此在转换完毕后需要分别修改 `ResNet50_vd_serving` 下的文件 `serving_server_conf.prototxt` 和 `ResNet50_vd_client` 下的文件 `serving_client_conf.prototxt`，将 `fetch_var` 中 `alias_name:` 后的字段改为 `prediction`，修改后的 `serving_server_conf.prototxt` 和 `serving_client_conf.prototxt` 如下所示:
  ```log
  feed_var {
    name: "inputs"
    alias_name: "inputs"
    is_lod_tensor: false
    feed_type: 1
    shape: 3
    shape: 224
    shape: 224
  }
  fetch_var {
    name: "save_infer_model/scale_0.tmp_1"
    alias_name: "prediction"
    is_lod_tensor: false
    fetch_type: 1
    shape: 1000
  }
  ```
<a name="3.2"></a>
### 3.2 服务部署和请求

paddleserving 目录包含了启动 pipeline 服务、C++ serving服务和发送预测请求的代码，主要包括：
```shell
__init__.py
classification_web_service.py # 启动pipeline服务端的脚本
config.yml                    # 启动pipeline服务的配置文件
pipeline_http_client.py       # http方式发送pipeline预测请求的脚本
pipeline_rpc_client.py        # rpc方式发送pipeline预测请求的脚本
readme.md                     # 分类模型服务化部署文档
run_cpp_serving.sh            # 启动C++ Serving部署的脚本
test_cpp_serving_client.py    # rpc方式发送C++ serving预测请求的脚本
```
<a name="3.2.1"></a>
#### 3.2.1 Python Serving

- 启动服务：
  ```shell
  # 启动服务，运行日志保存在 log.txt
  python3.7 classification_web_service.py &>log.txt &
  ```

- 发送请求：
  ```shell
  # 发送服务请求
  python3.7 pipeline_http_client.py
  ```
  成功运行后，模型预测的结果会打印在客户端中，如下所示：
  ```log
  {'err_no': 0, 'err_msg': '', 'key': ['label', 'prob'], 'value': ["['daisy']", '[0.9341402053833008]'], 'tensors': []}
  ```
- 关闭服务
如果服务程序在前台运行，可以按下`Ctrl+C`来终止服务端程序；如果在后台运行，可以使用kill命令关闭相关进程，也可以在启动服务程序的路径下执行以下命令来终止服务端程序：
  ```bash
  python3.7 -m paddle_serving_server.serve stop
  ```
  执行完毕后出现`Process stopped`信息表示成功关闭服务。

<a name="3.2.2"></a>
#### 3.2.2 C++ Serving

与Python Serving不同，C++ Serving客户端调用 C++ OP来预测，因此在启动服务之前，需要编译并安装 serving server包，并设置 `SERVING_BIN`。

- 编译并安装Serving server包
  ```shell
  # 进入工作目录
  cd PaddleClas/deploy/paddleserving
  # 一键编译安装Serving server、设置 SERVING_BIN
  source ./build_server.sh python3.7
  ```
  **注：**[build_server.sh](./build_server.sh#L55-L62)所设定的路径可能需要根据实际机器上的环境如CUDA、python版本等作一定修改，然后再编译。

- 修改客户端文件 `ResNet50_vd_client/serving_client_conf.prototxt` ，将 `feed_type:` 后的字段改为20，将第一个 `shape:` 后的字段改为1并删掉其余的 `shape` 字段。
  ```log
  feed_var {
    name: "inputs"
    alias_name: "inputs"
    is_lod_tensor: false
    feed_type: 20
    shape: 1
  }
  ```
- 修改 [`test_cpp_serving_client`](./test_cpp_serving_client.py) 的部分代码
  1. 修改 [`load_client_config`](./test_cpp_serving_client.py#L28) 处的代码，将 `load_client_config` 后的路径改为 `ResNet50_vd_client/serving_client_conf.prototxt` 。
  2. 修改 [`feed={"inputs": image}`](./test_cpp_serving_client.py#L45) 处的代码，将 `inputs` 改为与 `ResNet50_vd_client/serving_client_conf.prototxt` 中 `feed_var` 字段下面的 `name` 一致。由于部分模型client文件中的 `name` 为 `x` 而不是 `inputs` ，因此使用这些模型进行C++ Serving部署时需要注意这一点。

- 启动服务：
  ```shell
  # 启动服务， 服务在后台运行，运行日志保存在 nohup.txt
  # CPU部署
  bash run_cpp_serving.sh
  # GPU部署并指定0号卡
  bash run_cpp_serving.sh 0
  ```

- 发送请求：
  ```shell
  # 发送服务请求
  python3.7 test_cpp_serving_client.py
  ```
  成功运行后，模型预测的结果会打印在客户端中，如下所示：
  ```log
  prediction: daisy, probability: 0.9341399073600769
  ```
- 关闭服务：
  如果服务程序在前台运行，可以按下`Ctrl+C`来终止服务端程序；如果在后台运行，可以使用kill命令关闭相关进程，也可以在启动服务程序的路径下执行以下命令来终止服务端程序：
  ```bash
  python3.7 -m paddle_serving_server.serve stop
  ```
  执行完毕后出现`Process stopped`信息表示成功关闭服务。

## 4.FAQ

**Q1**： 发送请求后没有结果返回或者提示输出解码报错

**A1**： 启动服务和发送请求时不要设置代理，可以在启动服务前和发送请求前关闭代理，关闭代理的命令是：
```shell
unset https_proxy
unset http_proxy
```

**Q2**： 启动服务后没有任何反应

**A2**： 可以检查`config.yml`中`model_config`对应的路径是否存在，文件夹命名是否正确

更多的服务部署类型，如 `RPC 预测服务` 等，可以参考 Serving 的[github 官网](https://github.com/PaddlePaddle/Serving/tree/v0.9.0/examples)
