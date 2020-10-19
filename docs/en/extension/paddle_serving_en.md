# Model Service Deployment

## Overview
[Paddle Serving](https://github.com/PaddlePaddle/Serving) aims to help deep-learning researchers to easily deploy online inference services, supporting one-click deployment of industry, high concurrency and efficient communication between client and server and supporting multiple programming languages to develop clients.

Taking HTTP inference service deployment as an example to introduce how to use PaddleServing to deploy model services in PaddleClas.

## Serving Install

It is recommends to use docker to install and deploy the Serving environment in the Serving official website, first, you need to pull the docker environment and create Serving-based docker.

```shell
nvidia-docker pull hub.baidubce.com/paddlepaddle/serving:0.2.0-gpu
nvidia-docker run -p 9292:9292 --name test -dit hub.baidubce.com/paddlepaddle/serving:0.2.0-gpu
nvidia-docker exec -it test bash
```

In docker, you need to install some packages about Serving

```shell
pip install paddlepaddle-gpu
pip install paddle-serving-client
pip install paddle-serving-server-gpu
```

* If the installation speed is too slow, you can add `-i https://pypi.tuna.tsinghua.edu.cn/simple` following pip to speed up the process.

* If you want to deploy CPU service, you can install the cpu version of Serving, the command is as follow.

```shell
pip install paddle-serving-server
```

### Export Model

Exporting the Serving model using `tools/export_serving_model.py`, taking ResNet50_vd as an example, the command is as follow.

```shell
python tools/export_serving_model.py -m ResNet50_vd -p ./pretrained/ResNet50_vd_pretrained/ -o serving
```

finally, the client configures, model parameters and structure file will be saved in `ppcls_client_conf` and `ppcls_model`.


### Service Deployment and Request

* Using the following commands to start the Serving.

```shell
python tools/serving/image_service_gpu.py serving/ppcls_model workdir 9292
```

`serving/ppcls_model` is the address of the Serving model just saved, `workdir` is the work directory, and `9292` is the port of the service.


* Using the following script to send an identification request to the Serving and return the result.

```
python tools/serving/image_http_client.py  9292 ./docs/images/logo.png
```

`9292` is the port for sending the request, which is consistent with the Serving starting port, and `./docs/images/logo.png` is the test image, the final top1 label and probability are returned.

* For more Serving deployment, such RPC inference service, you can refer to the Serving official website: [https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/imagenet](https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/imagenet)
