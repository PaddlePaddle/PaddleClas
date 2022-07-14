English | [简体中文](../../zh_CN/inference_deployment/classification_serving_deploy.md)

# Classification model service deployment

## Table of contents

- [1 Introduction](#1-introduction)
- [2. Serving installation](#2-serving-installation)
- [3. Image Classification Service Deployment](#3-image-classification-service-deployment)
  - [3.1 Model conversion](#31-model-conversion)
  - [3.2 Service deployment and request](#32-service-deployment-and-request)
    - [3.2.1 Python Serving](#321-python-serving)
    - [3.2.2 C++ Serving](#322-c-serving)

<a name="1"></a>
## 1 Introduction

[Paddle Serving](https://github.com/PaddlePaddle/Serving) aims to help deep learning developers easily deploy online prediction services, support one-click deployment of industrial-grade service capabilities, high concurrency between client and server Efficient communication and support for developing clients in multiple programming languages.

This section takes the HTTP prediction service deployment as an example to introduce how to use PaddleServing to deploy the model service in PaddleClas. Currently, only Linux platform deployment is supported, and Windows platform is not currently supported.

<a name="2"></a>
## 2. Serving installation

The Serving official website recommends using docker to install and deploy the Serving environment. First, you need to pull the docker environment and create a Serving-based docker.

```shell
# start GPU docker
docker pull paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel
nvidia-docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel bash
nvidia-docker exec -it test bash

# start CPU docker
docker pull paddlepaddle/serving:0.7.0-devel
docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-devel bash
docker exec -it test bash
```

After entering docker, you need to install Serving-related python packages.
```shell
python3.7 -m pip install paddle-serving-client==0.7.0
python3.7 -m pip install paddle-serving-app==0.7.0
python3.7 -m pip install faiss-cpu==1.7.1post2

#If it is a CPU deployment environment:
python3.7 -m pip install paddle-serving-server==0.7.0 #CPU
python3.7 -m pip install paddlepaddle==2.2.0 # CPU

#If it is a GPU deployment environment
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post102 # GPU with CUDA10.2 + TensorRT6
python3.7 -m pip install paddlepaddle-gpu==2.2.0 # GPU with CUDA10.2

#Other GPU environments need to confirm the environment and then choose which one to execute
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post101 # GPU with CUDA10.1 + TensorRT6
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post112 # GPU with CUDA11.2 + TensorRT8
```

* If the installation speed is too slow, you can change the source through `-i https://pypi.tuna.tsinghua.edu.cn/simple` to speed up the installation process.
* For other environment configuration installation, please refer to: [Install Paddle Serving with Docker](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Install_EN.md)

<a name="3"></a>

## 3. Image Classification Service Deployment

The following takes the classic ResNet50_vd model as an example to introduce how to deploy the image classification service.

<a name="3.1"></a>
### 3.1 Model conversion

When using PaddleServing for service deployment, you need to convert the saved inference model into a Serving model.
- Go to the working directory:
  ```shell
  cd deploy/paddleserving
  ```
- Download and unzip the inference model for ResNet50_vd:
  ```shell
  # Download ResNet50_vd inference model
  wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar
  # Decompress the ResNet50_vd inference model
  tar xf ResNet50_vd_infer.tar
  ```
- Use the paddle_serving_client command to convert the downloaded inference model into a model format for easy server deployment:
  ```shell
  # Convert ResNet50_vd model
  python3.7 -m paddle_serving_client.convert \
  --dirname ./ResNet50_vd_infer/ \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --serving_server ./ResNet50_vd_serving/ \
  --serving_client ./ResNet50_vd_client/
  ```
  The specific meaning of the parameters in the above command is shown in the following table
    | parameter | type | default value | description |
    | --------- | ---- | ------------- | ----------- |  |--- |
  | `dirname` | str | - | The storage path of the model file to be converted. The program structure file and parameter file are saved in this directory. |
  | `model_filename` | str | None | The name of the file storing the model Inference Program structure that needs to be converted. If set to None, use `__model__` as the default filename |
  | `params_filename` | str | None | File name where all parameters of the model to be converted are stored. It needs to be specified if and only if all model parameters are stored in a single binary file. If the model parameters are stored in separate files, set it to None |
  | `serving_server` | str | `"serving_server"` | The storage path of the converted model files and configuration files. Default is serving_server |
  | `serving_client` | str | `"serving_client"` | The converted client configuration file storage path. Default is serving_client |

    After the ResNet50_vd inference model conversion is completed, there will be additional `ResNet50_vd_serving` and `ResNet50_vd_client` folders in the current folder, with the following structure:
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

- Serving provides the function of input and output renaming in order to be compatible with the deployment of different models. When different models are deployed in inference, you only need to modify the `alias_name` of the configuration file, and the inference deployment can be completed without modifying the code. Therefore, after the conversion, you need to modify the alias names in the files `serving_server_conf.prototxt` under `ResNet50_vd_serving` and `ResNet50_vd_client` respectively, and change the `alias_name` in `fetch_var` to `prediction`, the modified serving_server_conf.prototxt is as follows Show:
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
### 3.2 Service deployment and request

The paddleserving directory contains the code for starting the pipeline service, the C++ serving service and sending the prediction request, mainly including:
```shell
__init__.py
classification_web_service.py # Script to start the pipeline server
config.yml # Configuration file to start the pipeline service
pipeline_http_client.py # Script for sending pipeline prediction requests in http mode
pipeline_rpc_client.py # Script for sending pipeline prediction requests in rpc mode
readme.md # Classification model service deployment document
run_cpp_serving.sh # Start the C++ Serving departmentscript
test_cpp_serving_client.py # Script for sending C++ serving prediction requests in rpc mode
```
<a name="3.2.1"></a>
#### 3.2.1 Python Serving

- Start the service:
  ```shell
  # Start the service and save the running log in log.txt
  python3.7 classification_web_service.py &>log.txt &
  ```

- send request:
  ```shell
  # send service request
  python3.7 pipeline_http_client.py
  ```
  After a successful run, the results of the model prediction will be printed in the cmd window, and the results are as follows:
  ```log
  {'err_no': 0, 'err_msg': '', 'key': ['label', 'prob'], 'value': ["['daisy']", '[0.9341402053833008]'], 'tensors ': []}
  ```
- turn off the service
If the service program is running in the foreground, you can press `Ctrl+C` to terminate the server program; if it is running in the background, you can use the kill command to close related processes, or you can execute the following command in the path where the service program is started to terminate the server program:
  ```bash
  python3.7 -m paddle_serving_server.serve stop
  ```
  After the execution is completed, the `Process stopped` message appears, indicating that the service was successfully shut down.

<a name="3.2.2"></a>
#### 3.2.2 C++ Serving

Different from Python Serving, the C++ Serving client calls C++ OP to predict, so before starting the service, you need to compile and install the serving server package, and set `SERVING_BIN`.

- Compile and install the Serving server package
  ```shell
  # Enter the working directory
  cd PaddleClas/deploy/paddleserving
  # One-click compile and install Serving server, set SERVING_BIN
  source ./build_server.sh python3.7
  ```
  **Note: The path set by **[build_server.sh](./build_server.sh#L55-L62) may need to be modified according to the actual machine environment such as CUDA, python version, etc., and then compiled.

- Modify the client file `ResNet50_client/serving_client_conf.prototxt` , change the field after `feed_type:` to 20, change the field after the first `shape:` to 1 and delete the rest of the `shape` fields.
  ```log
  feed_var {
    name: "inputs"
    alias_name: "inputs"
    is_lod_tensor: false
    feed_type: 20
    shape: 1
  }
  ```
- Modify part of the code of [`test_cpp_serving_client`](./test_cpp_serving_client.py)
  1. Modify the [`feed={"inputs": image}`](./test_cpp_serving_client.py#L28) part of the code, and change the path after `load_client_config` to `ResNet50_client/serving_client_conf.prototxt` .
  2. Modify the [`feed={"inputs": image}`](./test_cpp_serving_client.py#L45) part of the code, and change `inputs` to be the same as the `feed_var` field in `ResNet50_client/serving_client_conf.prototxt` name` is the same. Since `name` in some model client files is `x` instead of `inputs` , you need to pay attention to this when using these models for C++ Serving deployment.

- Start the service:
  ```shell
  # Start the service, the service runs in the background, and the running log is saved in nohup.txt
  # CPU deployment
  sh run_cpp_serving.sh
  # GPU deployment and specify card 0
  sh run_cpp_serving.sh 0
  ```

- send request:
  ```shell
  # send service request
  python3.7 test_cpp_serving_client.py
  ```
  After a successful run, the results of the model prediction will be printed in the cmd window, and the results are as follows:
  ```log
  prediction: daisy, probability: 0.9341399073600769
  ```
- close the service:
  If the service program is running in the foreground, you can press `Ctrl+C` to terminate the server program; if it is running in the background, you can use the kill command to close related processes, or you can execute the following command in the path where the service program is started to terminate the server program:
  ```bash
  python3.7 -m paddle_serving_server.serve stop
  ```
  After the execution is completed, the `Process stopped` message appears, indicating that the service was successfully shut down.

##4.FAQ

**Q1**: No result is returned after the request is sent or an output decoding error is prompted

**A1**: Do not set the proxy when starting the service and sending the request. You can close the proxy before starting the service and sending the request. The command to close the proxy is:
```shell
unset https_proxy
unset http_proxy
```

**Q2**: nothing happens after starting the service

**A2**: You can check whether the path corresponding to `model_config` in `config.yml` exists, and whether the folder name is correct

For more service deployment types, such as `RPC prediction service`, you can refer to Serving's [github official website](https://github.com/PaddlePaddle/Serving/tree/v0.9.0/examples)
