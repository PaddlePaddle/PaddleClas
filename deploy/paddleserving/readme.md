# 模型服务化部署
- [1. 简介](#1)
- [2. Serving 安装](#2)
- [3. 图像分类服务部署](#3)
    - [3.1 模型转换](#3.1)
    - [3.2 服务部署和请求](#3.2)
- [4. 图像识别服务部署](#4)
  - [4.1 模型转换](#4.1)
  - [4.2 服务部署和请求](#4.2)
- [5. FAQ](#5)

<a name="1"></a>
## 1. 简介
[Paddle Serving](https://github.com/PaddlePaddle/Serving) 旨在帮助深度学习开发者轻松部署在线预测服务，支持一键部署工业级的服务能力、客户端和服务端之间高并发和高效通信、并支持多种编程语言开发客户端。

该部分以 HTTP 预测服务部署为例，介绍怎样在 PaddleClas 中使用 PaddleServing 部署模型服务。目前只支持 Linux 平台部署，暂不支持 Windows 平台。

<a name="2"></a>
## 2. Serving 安装

Serving 官网推荐使用 docker 安装并部署 Serving 环境。首先需要拉取 docker 环境并创建基于 Serving 的 docker。

```shell
docker pull paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel
nvidia-docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel bash
nvidia-docker exec -it test bash
```

进入 docker 后，需要安装 Serving 相关的 python 包。

```shell
pip3 install paddle-serving-client==0.7.0
pip3 install paddle-serving-server==0.7.0 # CPU
pip3 install paddle-serving-app==0.7.0
pip3 install paddle-serving-server-gpu==0.7.0.post102 #GPU with CUDA10.2 + TensorRT6
# 其他GPU环境需要确认环境再选择执行哪一条
pip3 install paddle-serving-server-gpu==0.7.0.post101 # GPU with CUDA10.1 + TensorRT6
pip3 install paddle-serving-server-gpu==0.7.0.post112 # GPU with CUDA11.2 + TensorRT8
```

* 如果安装速度太慢，可以通过 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 更换源，加速安装过程。
* 其他环境配置安装请参考: [使用Docker安装Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Install_CN.md)

* 如果希望部署 CPU 服务，可以安装 serving-server 的 cpu 版本，安装命令如下。

```shell
pip install paddle-serving-server
```

## 3. 图像分类服务部署
<a name="3.1"></a>
### 3.1 模型转换
使用 PaddleServing 做服务化部署时，需要将保存的 inference 模型转换为 Serving 模型。下面以经典的 ResNet50_vd 模型为例，介绍如何部署图像分类服务。
- 进入工作目录：
```shell
cd deploy/paddleserving
```
- 下载 ResNet50_vd 的 inference 模型：
```shell
# 下载并解压 ResNet50_vd 模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar && tar xf ResNet50_vd_infer.tar
```
- 用 paddle_serving_client 把下载的 inference 模型转换成易于 Server 部署的模型格式：
```
# 转换 ResNet50_vd 模型
python3 -m paddle_serving_client.convert --dirname ./ResNet50_vd_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./ResNet50_vd_serving/ \
                                         --serving_client ./ResNet50_vd_client/
```
ResNet50_vd 推理模型转换完成后，会在当前文件夹多出 `ResNet50_vd_serving` 和 `ResNet50_vd_client` 的文件夹，具备如下格式：
```
|- ResNet50_vd_server/
  |- inference.pdiparams  
  |- inference.pdmodel
  |- serving_server_conf.prototxt  
  |- serving_server_conf.stream.prototxt
|- ResNet50_vd_client
  |- serving_client_conf.prototxt  
  |- serving_client_conf.stream.prototxt
```
得到模型文件之后，需要修改 serving_server_conf.prototxt 中的 alias 名字：将 `fetch_var` 中的 `alias_name` 改为 `prediction`

**备注**:  Serving 为了兼容不同模型的部署，提供了输入输出重命名的功能。这样，不同的模型在推理部署时，只需要修改配置文件的 alias_name 即可，无需修改代码即可完成推理部署。
修改后的 serving_server_conf.prototxt 如下所示:
```
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
  is_lod_tensor: true
  fetch_type: 1
  shape: -1
}
```
<a name="3.2"></a>
### 3.2 服务部署和请求
paddleserving 目录包含了启动 pipeline 服务和发送预测请求的代码，包括：
```shell
__init__.py
config.yml                 # 启动服务的配置文件
pipeline_http_client.py    # http方式发送pipeline预测请求的脚本
pipeline_rpc_client.py     # rpc方式发送pipeline预测请求的脚本
classification_web_service.py    # 启动pipeline服务端的脚本
```

- 启动服务：
```shell
# 启动服务，运行日志保存在 log.txt
python3 classification_web_service.py &>log.txt &
```
成功启动服务后，log.txt 中会打印类似如下日志
![](./imgs/start_server.png)

- 发送请求：
```shell
# 发送服务请求
python3 pipeline_http_client.py
```
成功运行后，模型预测的结果会打印在 cmd 窗口中，结果示例为：
![](./imgs/results.png)

<a name="4"></a>
## 4.图像识别服务部署
使用 PaddleServing 做服务化部署时，需要将保存的 inference 模型转换为 Serving 模型。 下面以 PP-ShiTu 中的超轻量图像识别模型为例，介绍图像识别服务的部署。
<a name="4.1"></a>
## 4.1 模型转换
- 下载通用检测 inference 模型和通用识别 inference 模型
```
cd deploy
# 下载并解压通用识别模型
wget -P models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
cd models
tar -xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
# 下载并解压通用检测模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
```
- 转换识别 inference 模型为 Serving 模型：
```
# 转换识别模型
python3 -m paddle_serving_client.convert --dirname ./general_PPLCNet_x2_5_lite_v1.0_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./general_PPLCNet_x2_5_lite_v1.0_serving/ \
                                         --serving_client ./general_PPLCNet_x2_5_lite_v1.0_client/
```
识别推理模型转换完成后，会在当前文件夹多出 `general_PPLCNet_x2_5_lite_v1.0_serving/` 和 `general_PPLCNet_x2_5_lite_v1.0_client/` 的文件夹。修改 `general_PPLCNet_x2_5_lite_v1.0_serving/` 目录下的 serving_server_conf.prototxt 中的 alias 名字： 将 `fetch_var` 中的 `alias_name` 改为 `features`。
修改后的 serving_server_conf.prototxt 内容如下：
```
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
  is_lod_tensor: true
  fetch_type: 1
  shape: -1
}
```
- 转换通用检测 inference 模型为 Serving 模型：
```
# 转换通用检测模型
python3 -m paddle_serving_client.convert --dirname ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/ \
                                         --serving_client ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
```
检测 inference 模型转换完成后，会在当前文件夹多出 `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/` 和 `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/` 的文件夹。

**注意:** 此处不需要修改 `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/` 目录下的 serving_server_conf.prototxt 中的 alias 名字。

- 下载并解压已经构建后的检索库 index
```
cd ../
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar && tar -xf drink_dataset_v1.0.tar
```
<a name="4.2"></a>
## 4.2 服务部署和请求
**注意:** 识别服务涉及到多个模型，出于性能考虑采用 PipeLine 部署方式。Pipeline 部署方式当前不支持 windows 平台。
- 进入到工作目录
```shell
cd ./deploy/paddleserving/recognition
```
paddleserving 目录包含启动 pipeline 服务和发送预测请求的代码，包括：
```
__init__.py
config.yml                    # 启动服务的配置文件
pipeline_http_client.py       # http方式发送pipeline预测请求的脚本
pipeline_rpc_client.py        # rpc方式发送pipeline预测请求的脚本
recognition_web_service.py    # 启动pipeline服务端的脚本
```
- 启动服务：
```
# 启动服务，运行日志保存在 log.txt
python3 recognition_web_service.py &>log.txt &
```
成功启动服务后，log.txt 中会打印类似如下日志
![](./imgs/start_server_shitu.png)

- 发送请求：
```
python3 pipeline_http_client.py
```
成功运行后，模型预测的结果会打印在 cmd 窗口中，结果示例为：
![](./imgs/results_shitu.png)

<a name="5"></a>
## 5.FAQ
**Q1**： 发送请求后没有结果返回或者提示输出解码报错

**A1**： 启动服务和发送请求时不要设置代理，可以在启动服务前和发送请求前关闭代理，关闭代理的命令是：
```
unset https_proxy
unset http_proxy
```

更多的服务部署类型，如 `RPC 预测服务` 等，可以参考 Serving 的[github 官网](https://github.com/PaddlePaddle/Serving/tree/v0.7.0/examples)
