English|[Chinese](../../zh_CN/inference_deployment/paddle_serving_deploy.md)
# Model Service deployment
--------
## Catalogue
- [1. Introduction](#1)
- [2. Installation of Serving](#2)
- [3. Service Deployment for Image Classification](#3)
    - [3.1 Model Transformation](#3.1)
    - [3.2 Service Deployment and Request](#3.2)
        - [3.2.1 Python Serving](#3.2.1)
        - [3.2.2 C++ Serving](#3.2.2)
- [4. Service Deployment for  Image Recognition](#4)
    - [4.1 Model Transformation](#4.1)
    - [4.2 Service Deployment and Request](#4.2)
        - [4.2.1 Python Serving](#4.2.1)
        - [4.2.2 C++ Serving](#4.2.2)
- [5. FAQ](#5)

<a name="1"></a>
## 1 Introduction
[Paddle Serving](https://github.com/PaddlePaddle/Serving) aims to help deep learning developers easily deploy online prediction services, support one-click deployment of industrial-grade service capabilities, high concurrency between client and server Efficient communication and support for developing clients in multiple programming languages.

This section takes the HTTP prediction service deployment as an example to introduce how to use PaddleServing to deploy the model service in PaddleClas. Currently, only Linux platform deployment is supported, and Windows platform is not currently supported.

<a name="2"></a>
## 2. Installation of Serving
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
* For other environment configuration installation, please refer to: [Install Paddle Serving with Docker](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Install_CN.md)

<a name="3"></a>

## 3. Service Deployment for Image Classification

The following takes the classic ResNet50_vd model as an example to introduce how to deploy the image classification service.

<a name="3.1"></a>
### 3.1 Model Transformation
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
  | ----------------- | ---- | ------------------ | ------------------------------------------------------------ |
  | `dirname` | str | - | The storage path of the model file to be converted. The program structure file and parameter file are saved in this directory. |
  | `model_filename` | str | None | The name of the file storing the model Inference Program structure that needs to be converted. If set to None, use `__model__` as the default filename |
  | `params_filename` | str | None | File name where all parameters of the model to be converted are stored. It needs to be specified if and only if all model parameters are stored in a single binary file. If the model parameters are stored in separate files, set it to None |
  | `serving_server` | str | `"serving_server"` | The storage path of the converted model files and configuration files. Default is serving_server |
  | `serving_client` | str | `"serving_client"` | The converted client configuration file storage path. Default is serving_client |

    After the ResNet50_vd inference model is converted, there will be additional `ResNet50_vd_serving` and `ResNet50_vd_client` folders in the current folder, with the following structure:
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
### 3.2 Service Deployment and Request
The paddleserving directory contains the code to start the pipeline service, C++ serving service and send prediction requests, including:
```shell
__init__.py
classification_web_service.py # Script to start the pipeline server
config.yml # Configuration file to start the pipeline service
pipeline_http_client.py # Script for sending pipeline prediction requests in http mode
pipeline_rpc_client.py # Script for sending pipeline prediction requests in rpc mode
run_cpp_serving.sh# Script to start C++ Serving deployment
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
  ```
  {'err_no': 0, 'err_msg': '', 'key': ['label', 'prob'], 'value': ["['daisy']", '[0.9341402053833008]'], 'tensors ': []}
  ```

<a name="3.2.2"></a>
#### 3.2.2 C++ Serving
- Start the service:
  ```shell
  # Start the service, the service runs in the background, and the running log is saved in nohup.txt
  sh run_cpp_serving.sh
  ```

- send request:
  ```shell
  # send service request
  python3.7 test_cpp_serving_client.py
  ```
  After a successful run, the results of the model prediction will be printed in the cmd window, and the results are as follows:
  ```
  prediction: daisy, probability: 0.9341399073600769
  ```

<a name="4"></a>
## 4. Service Deployment for Image Recognition
In addition to the single-model deployment method introduced in [Chapter 3 Service Deployment for Image Classification](#3), we will introduce how to use the detection + classification model to complete the multi-model **image recognition service deployment**
When using PaddleServing for image recognition service deployment, **need to convert multiple saved inference models to Serving models**. The following takes the ultra-lightweight image recognition model in PP-ShiTu as an example to introduce the deployment of image recognition services.
<a name="4.1"></a>
### 4.1 Model Transformation
- Go to the working directory:
  ```shell
  cd deploy/
  ```
- Download generic detection inference model and generic recognition inference model
  ```shell
  # Create and enter the models folder
  mkdir models
  cd models
  # Download and unzip the generic recognition model
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
  tar -xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
  # Download and unzip the generic detection model
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
  tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
  ```
- Convert the generic recognition inference model to the Serving model:
  ```shell
  # Convert the generic recognition model
  python3.7 -m paddle_serving_client.convert \
  --dirname ./general_PPLCNet_x2_5_lite_v1.0_infer/ \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --serving_server ./general_PPLCNet_x2_5_lite_v1.0_serving/ \
  --serving_client ./general_PPLCNet_x2_5_lite_v1.0_client/
  ```
  The meaning of the parameters of the above command is the same as [#3.1 Model Transformation](#3.1)
  After the transformation of the general recognition inference model is completed, there will be additional `general_PPLCNet_x2_5_lite_v1.0_serving/` and `general_PPLCNet_x2_5_lite_v1.0_client/` folders in the current folder, with the following structure:
    ```shell
  ├── general_PPLCNet_x2_5_lite_v1.0_serving/
  │ ├── inference.pdiparams
  │ ├── inference.pdmodel
  │ ├── serving_server_conf.prototxt
  │ └── serving_server_conf.stream.prototxt
  │
  └── general_PPLCNet_x2_5_lite_v1.0_client/
        ├── serving_client_conf.prototxt
        └── serving_client_conf.stream.prototxt
  ```

- Modify the alias names in `serving_server_conf.prototxt` in `general_PPLCNet_x2_5_lite_v1.0_serving/` and `general_PPLCNet_x2_5_lite_v1.0_client/` directories respectively: change `alias_name` in `fetch_var` to `features`.
  The modified `serving_server_conf.prototxt` content is as follows:
  ```shell
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
- Convert general detection inference model to Serving model:
  ```shell
  # Convert generic detection model
  python3.7 -m paddle_serving_client.convert --dirname ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer/ \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --serving_server ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/ \
  --serving_client ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
  ```
  The meaning of the parameters of the above command is the same as [#3.1 Model Transformation](#3.1)

  After the general detection inference model transformation is completed, there will be additional folders `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/` and `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/` in the current folder, with the following structure:
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

  **Note:** There is no need to modify the alias name in serving_server_conf.prototxt under `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/` directory.

- Download and unzip the built retrieval library index
    ```shell
    # Go back to the deploy directory
    cd ../
    # Download the built retrieval library index
    wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar
    # Decompress the built retrieval library index
    tar -xf drink_dataset_v1.0.tar
    ```
<a name="4.2"></a>
### 4.2 Service Deployment and Request
**Note:** The identification service involves multiple models, and the PipeLine deployment method is used for performance reasons. The Pipeline deployment method currently does not support the windows platform.
- go to the working directory
  ```shell
  cd ./deploy/paddleserving/recognition
  ```
  The paddleserving directory contains code to start the Python Pipeline service, the C++ Serving service, and send prediction requests, including:
  ```shell
  __init__.py
  config.yml # The configuration file to start the python pipeline service
  pipeline_http_client.py # Script for sending pipeline prediction requests in http mode
  pipeline_rpc_client.py # Script for sending pipeline prediction requests in rpc mode
  recognition_web_service.py # Script to start the pipeline server
  run_cpp_serving.sh # Script to start C++ Pipeline Serving deployment
  test_cpp_serving_client.py # Script for sending C++ Pipeline serving prediction requests by rpc
  ```

<a name="4.2.1"></a>
#### 4.2.1 Python Serving
- Start the service:
  ```shell
  # Start the service, run the logSave in log.txt
  python3.7 recognition_web_service.py &>log.txt &
  ```

- send request:
  ```shell
  python3.7 pipeline_http_client.py
  ```
  After a successful run, the results of the model prediction will be printed in the cmd window, and the results are as follows:
  ```log
  {'err_no': 0, 'err_msg': '', 'key': ['result'], 'value': ["[{'bbox': [345, 95, 524, 576], 'rec_docs': 'Red Bull-Enhanced', 'rec_scores': 0.79903316}]"], 'tensors': []}
  ```

<a name="4.2.2"></a>
#### 4.2.2 C++ Serving
- Start the service:
  ```shell
  # Start the service: Here, the subject detection and feature extraction services will be started at the same time in the background, and the port numbers are 9293 and 9294 respectively;
  # The running logs are saved in log_mainbody_detection.txt and log_feature_extraction.txt respectively
  sh run_cpp_serving.sh
  ```

- send request:
  ```shell
  # send service request
  python3.7 test_cpp_serving_client.py
  ```
  After a successful run, the results of the model predictions are printed in the cmd window, and the results are as follows:
  ```log
  [{'bbox': [345, 95, 524, 586], 'rec_docs': 'Red Bull-Enhanced', 'rec_scores': 0.8016462}]
  ```

<a name="5"></a>
## 5. FAQ

**Q1**: No result is returned after the request is sent or an output decoding error is prompted

**A1**: Do not set the proxy when starting the service and sending the request. You can close the proxy before starting the service and sending the request. The command to close the proxy is:
```shell
unset https_proxy
unset http_proxy
```

**Q2**: nothing happens after starting the service

**A2**: You can check whether the path corresponding to `model_config` in `config.yml` exists, and whether the folder name is correct


For more service deployment types, such as `RPC prediction service`, you can refer to Serving's [github official website](https://github.com/PaddlePaddle/Serving/tree/v0.9.0/examples)
