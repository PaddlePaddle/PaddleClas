简体中文 | [English](../../en/inference_deployment/recognition_serving_deploy_en.md)

# 识别模型服务化部署

## 目录

- [1. 简介](#1-简介)
- [2. Serving 安装](#2-serving-安装)
- [3. 图像识别服务部署](#3-图像识别服务部署)
  - [3.1 模型转换](#31-模型转换)
  - [3.2 服务部署和请求](#32-服务部署和请求)
    - [3.2.1 Python Serving](#321-python-serving)
    - [3.2.2 C++ Serving](#322-c-serving)
- [4. FAQ](#4-faq)

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
## 3. 图像识别服务部署

使用 PaddleServing 做图像识别服务化部署时，**需要将保存的多个 inference 模型都转换为 Serving 模型**。 下面以 PP-ShiTu 中的超轻量图像识别模型为例，介绍图像识别服务的部署。
<a name="3.1"></a>

### 3.1 模型转换

- 进入工作目录：
  ```shell
  cd deploy/
  ```
- 下载通用检测 inference 模型和通用识别 inference 模型
  ```shell
  # 创建并进入models文件夹
  mkdir models
  cd models
  # 下载并解压通用识别模型
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
  tar -xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
  # 下载并解压通用检测模型
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
  tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
  ```
- 转换通用识别 inference 模型为 Serving 模型：
  ```shell
  # 转换通用识别模型
  python3.7 -m paddle_serving_client.convert \
  --dirname ./general_PPLCNet_x2_5_lite_v1.0_infer/ \
  --model_filename inference.pdmodel  \
  --params_filename inference.pdiparams \
  --serving_server ./general_PPLCNet_x2_5_lite_v1.0_serving/ \
  --serving_client ./general_PPLCNet_x2_5_lite_v1.0_client/
  ```
  上述命令的参数含义与[#3.1 模型转换](#3.1)相同
  通用识别 inference 模型转换完成后，会在当前文件夹多出 `general_PPLCNet_x2_5_lite_v1.0_serving/` 和 `general_PPLCNet_x2_5_lite_v1.0_client/` 的文件夹，具备如下结构：
    ```shell
    ├── general_PPLCNet_x2_5_lite_v1.0_serving/
    │   ├── inference.pdiparams
    │   ├── inference.pdmodel
    │   ├── serving_server_conf.prototxt
    │   └── serving_server_conf.stream.prototxt
    │
    └── general_PPLCNet_x2_5_lite_v1.0_client/
          ├── serving_client_conf.prototxt
          └── serving_client_conf.stream.prototxt
    ```
- 转换通用检测 inference 模型为 Serving 模型：
  ```shell
  # 转换通用检测模型
  python3.7 -m paddle_serving_client.convert --dirname ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer/ \
  --model_filename inference.pdmodel  \
  --params_filename inference.pdiparams \
  --serving_server ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/ \
  --serving_client ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
  ```
  上述命令的参数含义与[#3.1 模型转换](#3.1)相同

  识别推理模型转换完成后，会在当前文件夹多出 `general_PPLCNet_x2_5_lite_v1.0_serving/` 和 `general_PPLCNet_x2_5_lite_v1.0_client/` 的文件夹。分别修改 `general_PPLCNet_x2_5_lite_v1.0_serving/` 和 `general_PPLCNet_x2_5_lite_v1.0_client/` 目录下的 `serving_server_conf.prototxt` 中的 `alias` 名字： 将 `fetch_var` 中的 `alias_name` 改为 `features`。 修改后的 `serving_server_conf.prototxt` 内容如下

  ```log
  feed_var {
  name: "x"
  alias_name: "x"
  is_lod_tensor: false
  feed_type: 1
  shape: 3
  shape: 224
  shape: 224
  }
  fetch_var {
    name: "save_infer_model/scale_0.tmp_1"
    alias_name: "features"
    is_lod_tensor: false
    fetch_type: 1
    shape: 512
  }
  ```
  通用检测 inference 模型转换完成后，会在当前文件夹多出 `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/` 和 `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/` 的文件夹，具备如下结构：
    ```shell
    ├── picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/
    │   ├── inference.pdiparams
    │   ├── inference.pdmodel
    │   ├── serving_server_conf.prototxt
    │   └── serving_server_conf.stream.prototxt
    │
    └── picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
          ├── serving_client_conf.prototxt
          └── serving_client_conf.stream.prototxt
    ```
  上述命令中参数具体含义如下表所示
    | 参数              | 类型 | 默认值             | 描述                                                         |
  | ----------------- | ---- | ------------------ | ------------------------------------------------------------ |
  | `dirname`         | str  | -                  | 需要转换的模型文件存储路径，Program结构文件和参数文件均保存在此目录。 |
  | `model_filename`  | str  | None               | 存储需要转换的模型Inference Program结构的文件名称。如果设置为None，则使用 `__model__` 作为默认的文件名 |
  | `params_filename` | str  | None               | 存储需要转换的模型所有参数的文件名称。当且仅当所有模型参数被保>存在一个单独的二进制文件中，它才需要被指定。如果模型参数是存储在各自分离的文件中，设置它的值为None |
  | `serving_server`  | str  | `"serving_server"` | 转换后的模型文件和配置文件的存储路径。默认值为serving_server |
  | `serving_client`  | str  | `"serving_client"` | 转换后的客户端配置文件存储路径。默认值为serving_client       |

- 下载并解压已经构建后完成的检索库 index
    ```shell
    # 回到deploy目录
    cd ../
    # 下载构建完成的检索库 index
    wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar
    # 解压构建完成的检索库 index
    tar -xf drink_dataset_v1.0.tar
    ```
<a name="3.2"></a>
### 3.2 服务部署和请求

**注意：** 识别服务涉及到多个模型，出于性能考虑采用 PipeLine 部署方式。Pipeline 部署方式当前不支持 windows 平台。
- 进入到工作目录
  ```shell
  cd ./deploy/paddleserving/recognition
  ```
  paddleserving 目录包含启动 Python Pipeline 服务、C++ Serving 服务和发送预测请求的代码，包括：
  ```shell
  __init__.py
  config.yml                  # 启动python pipeline服务的配置文件
  pipeline_http_client.py     # http方式发送pipeline预测请求的脚本
  pipeline_rpc_client.py      # rpc方式发送pipeline预测请求的脚本
  recognition_web_service.py  # 启动pipeline服务端的脚本
  readme.md                   # 识别模型服务化部署文档
  run_cpp_serving.sh          # 启动C++ Pipeline Serving部署的脚本
  test_cpp_serving_client.py  # rpc方式发送C++ Pipeline serving预测请求的脚本
  ```

<a name="3.2.1"></a>
#### 3.2.1 Python Serving

- 启动服务：
  ```shell
  # 启动服务，运行日志保存在 log.txt
  python3.7 recognition_web_service.py &>log.txt &
  ```

- 发送请求：
  ```shell
  python3.7 pipeline_http_client.py
  ```
  成功运行后，模型预测的结果会打印在客户端中，如下所示：
  ```log
  {'err_no': 0, 'err_msg': '', 'key': ['result'], 'value': ["[{'bbox': [345, 95, 524, 576], 'rec_docs': '红牛-强化型', 'rec_scores': 0.79903316}]"], 'tensors': []}
  ```

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
  **注：**[build_server.sh](../build_server.sh#L55-L62)所设定的路径可能需要根据实际机器上的环境如CUDA、python版本等作一定修改，然后再编译。

- C++ Serving使用的输入输出格式与Python不同，因此需要执行以下命令，将4个文件复制到下的文件覆盖掉[3.1](#31-模型转换)得到文件夹中的对应4个prototxt文件。
  ```shell
  # 进入PaddleClas/deploy目录
  cd PaddleClas/deploy/

  # 覆盖prototxt文件
  \cp ./paddleserving/recognition/preprocess/general_PPLCNet_x2_5_lite_v1.0_serving/*.prototxt ./models/general_PPLCNet_x2_5_lite_v1.0_serving/
  \cp ./paddleserving/recognition/preprocess/general_PPLCNet_x2_5_lite_v1.0_client/*.prototxt ./models/general_PPLCNet_x2_5_lite_v1.0_client/
  \cp ./paddleserving/recognition/preprocess/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/*.prototxt ./models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
  \cp ./paddleserving/recognition/preprocess/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/*.prototxt ./models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/
  ```

- 启动服务：
  ```shell
  # 进入工作目录
  cd PaddleClas/deploy/paddleserving/recognition

  # 端口号默认为9400；运行日志默认保存在 log_PPShiTu.txt 中
  # CPU部署
  bash run_cpp_serving.sh
  # GPU部署，并指定第0号卡
  bash run_cpp_serving.sh 0
  ```

- 发送请求：
  ```shell
  # 发送服务请求
  python3.7 test_cpp_serving_client.py
  ```
  成功运行后，模型预测的结果会打印在客户端中，如下所示：
  ```log
  WARNING: Logging before InitGoogleLogging() is written to STDERR
  I0614 03:01:36.273097  6084 naming_service_thread.cpp:202] brpc::policy::ListNamingService("127.0.0.1:9400"): added 1
  I0614 03:01:37.393564  6084 general_model.cpp:490] [client]logid=0,client_cost=1107.82ms,server_cost=1101.75ms.
  [{'bbox': [345, 95, 524, 585], 'rec_docs': '红牛-强化型', 'rec_scores': 0.8073724}]
  ```

- 关闭服务
如果服务程序在前台运行，可以按下`Ctrl+C`来终止服务端程序；如果在后台运行，可以使用kill命令关闭相关进程，也可以在启动服务程序的路径下执行以下命令来终止服务端程序：
  ```bash
  python3.7 -m paddle_serving_server.serve stop
  ```
  执行完毕后出现`Process stopped`信息表示成功关闭服务。

<a name="4"></a>
## 4. FAQ

**Q1**： 发送请求后没有结果返回或者提示输出解码报错

**A1**： 启动服务和发送请求时不要设置代理，可以在启动服务前和发送请求前关闭代理，关闭代理的命令是：
```shell
unset https_proxy
unset http_proxy
```
**Q2**： 启动服务后没有任何反应

**A2**： 可以检查`config.yml`中`model_config`对应的路径是否存在，文件夹命名是否正确

更多的服务部署类型，如 `RPC 预测服务` 等，可以参考 Serving 的[github 官网](https://github.com/PaddlePaddle/Serving/tree/v0.9.0/examples)
