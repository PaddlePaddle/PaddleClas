# 模型服务化部署

## 1. 简介
[Paddle Serving](https://github.com/PaddlePaddle/Serving) 旨在帮助深度学习开发者轻松部署在线预测服务，支持一键部署工业级的服务能力、客户端和服务端之间高并发和高效通信、并支持多种编程语言开发客户端。

该部分以 HTTP 预测服务部署为例，介绍怎样在 PaddleClas 中使用 PaddleServing 部署模型服务。


## 2. Serving安装

Serving 官网推荐使用 docker 安装并部署 Serving 环境。首先需要拉取 docker 环境并创建基于 Serving 的 docker。

```shell
nvidia-docker pull hub.baidubce.com/paddlepaddle/serving:0.2.0-gpu
nvidia-docker run -p 9292:9292 --name test -dit hub.baidubce.com/paddlepaddle/serving:0.2.0-gpu
nvidia-docker exec -it test bash
```

进入 docker 后，需要安装 Serving 相关的 python 包。

```shell
pip install paddlepaddle-gpu
pip install paddle-serving-client
pip install paddle-serving-server-gpu
```

* 如果安装速度太慢，可以通过 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 更换源，加速安装过程。

* 如果希望部署 CPU 服务，可以安装 serving-server 的 cpu 版本，安装命令如下。

```shell
pip install paddle-serving-server
```

## 3. 导出模型

使用 `tools/export_serving_model.py` 脚本导出 Serving 模型，以 `ResNet50_vd` 为例，使用方法如下。

```shell
python tools/export_serving_model.py -m ResNet50_vd -p ./pretrained/ResNet50_vd_pretrained/ -o serving
```

最终在 serving 文件夹下会生成 `ppcls_client_conf` 与 `ppcls_model` 两个文件夹，分别存储了 client 配置、模型参数与结构文件。


## 4. 服务部署与请求

* 使用下面的方式启动 Serving 服务。

```shell
python tools/serving/image_service_gpu.py serving/ppcls_model workdir 9292
```

其中 `serving/ppcls_model` 为刚才保存的 Serving 模型地址，`workdir` 为工作目录，`9292` 为服务的端口号。


* 使用下面的脚本向 Serving 服务发送识别请求，并返回结果。

```
python tools/serving/image_http_client.py  9292 ./docs/images/logo.png
```

`9292` 为发送请求的端口号，需要与服务启动时的端口号保持一致，`./docs/images/logo.png` 为待识别的图像文件。最终返回 Top1 识别结果的类别 ID 以及概率值。

* 更多的服务部署类型，如 `RPC预测服务` 等，可以参考 Serving 的 github 官网：[https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/imagenet](https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/imagenet)
