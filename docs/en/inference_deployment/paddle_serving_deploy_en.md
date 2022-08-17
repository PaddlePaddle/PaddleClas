# Model Service Deployment

## Catalogue

- [1. Introduction](#1)
- [2. Installation of Serving](#2)
- [3. Service Deployment for Image Classification](#3)
  - [3.1 Model Conversion](#3.1)
  - [3.2 Service Deployment and Request](#3.2)
- [4. Service Deployment for Image Recognition](#4)
  - [4.1 Model Conversion](#4.1)
  - [4.2 Service Deployment and Request](#4.2)
- [5. FAQ](#5)

<a name="1"></a>
## 1. Introduction

[Paddle Serving](https://github.com/PaddlePaddle/Serving) is designed to provide easy deployment of on-line prediction services for deep learning developers, it supports one-click deployment of industrial-grade services, highly concurrent and efficient communication between client and server, and multiple programming languages for client development.

This section, exemplified by HTTP deployment of prediction service, describes how to deploy model services in PaddleClas with PaddleServing. Currently, only deployment on Linux platform is supported. Windows platform is not supported.

<a name="2"></a>
## 2. Installation of Serving

It is officially recommended to use docker for the installation and environment deployment of Serving. First, pull the docker and create a Serving-based one.

```shell
docker pull paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel
nvidia-docker run -p 9292:9292 --name test -dit paddlepaddle/serving:0.7.0-cuda10.2-cudnn7-devel bash
nvidia-docker exec -it test bash
```

Once you are in docker,  install the Serving-related python packages.

```shell
python3.7 -m pip install paddle-serving-client==0.7.0
python3.7 -m pip install paddle-serving-server==0.7.0 # CPU
python3.7 -m pip install paddle-serving-app==0.7.0
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post102 #GPU with CUDA10.2 + TensorRT6
# For other GPU environemnt, confirm the environment before choosing which one to execute
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post101 # GPU with CUDA10.1 + TensorRT6
python3.7 -m pip install paddle-serving-server-gpu==0.7.0.post112 # GPU with CUDA11.2 + TensorRT8
```

- Speed up the installation process by replacing the source with `-i https://pypi.tuna.tsinghua.edu.cn/simple`.
- For other environment configuration and installation, please refer to [Install Paddle Serving using docker](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Install_EN.md)
- To deploy CPU services, please install the CPU version of serving-server with the following command.

```shell
python3.7 -m pip install paddle-serving-server
```

<a name="3"></a>
## 3. Service Deployment for Image Classification

<a name="3.1"></a>
## 3. Image Classification Service Deployment

The following takes the classic ResNet50_vd model as an example to introduce how to deploy the image classification service.

<a name="3.1"></a>
### 3.1 Model Conversion

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
    | --------- | ---- | ------------- | ----------- |
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
  **Note: The path set by **[build_server.sh](../../../deploy/paddleserving/build_server.sh#L55-L62) may need to be modified according to the actual machine environment such as CUDA, python version, etc., and then compiled; if you encounter a non-network error during the execution of build_server.sh, you can manually copy the commands in the script to the terminal for execution.

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

- Modify part of the code of [`test_cpp_serving_client`](../../../deploy/paddleserving/test_cpp_serving_client.py)
  1. Modify the [`feed={"inputs": image}`](../../../deploy/paddleserving/test_cpp_serving_client.py#L28) part of the code, and change the path after `load_client_config` to `ResNet50_client/serving_client_conf.prototxt` .
  2. Modify the [`feed={"inputs": image}`](../../../deploy/paddleserving/test_cpp_serving_client.py#L45) part of the code, and change `inputs` to be the same as the `feed_var` field in `ResNet50_client/serving_client_conf.prototxt` name` is the same. Since `name` in some model client files is `x` instead of `inputs` , you need to pay attention to this when using these models for C++ Serving deployment.

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


<a name="4"></a>
## 4. Image recognition service deployment

When using PaddleServing for image recognition service deployment, **need to convert multiple saved inference models to Serving models**. The following takes the ultra-lightweight image recognition model in PP-ShiTu as an example to introduce the deployment of image recognition services.
<a name="4.1"></a>
### 4.1 Model Conversion

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
  The meaning of the parameters of the above command is the same as [#3.1 Model Conversion](#3.1)

  After the recognition inference model is converted, there will be additional folders `general_PPLCNet_x2_5_lite_v1.0_serving/` and `general_PPLCNet_x2_5_lite_v1.0_client/` in the current folder. Modify the name of `alias` in `serving_server_conf.prototxt` in `general_PPLCNet_x2_5_lite_v1.0_serving/` and `general_PPLCNet_x2_5_lite_v1.0_client/` directories respectively: Change `alias_name` in `fetch_var` to `features`. The content of the modified `serving_server_conf.prototxt` is as follows

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

  After the conversion of the general recognition inference model is completed, there will be additional `general_PPLCNet_x2_5_lite_v1.0_serving/` and `general_PPLCNet_x2_5_lite_v1.0_client/` folders in the current folder, with the following structure:
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
- Convert general detection inference model to Serving model:
  ```shell
  # Convert generic detection model
  python3.7 -m paddle_serving_client.convert --dirname ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer/ \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --serving_server ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/ \
  --serving_client ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
  ```
  The meaning of the parameters of the above command is the same as [#3.1 Model Conversion](#3.1)

  After the conversion of the general detection inference model is completed, there will be additional folders `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/` and `picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/` in the current folder, with the following structure:
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
  The specific meaning of the parameters in the above command is shown in the following table
    | parameter         | type | default value      | description                                         |
    | ----------------- | ---- | ------------------ | ----------------------------------------------------- |
    | `dirname`         | str  | -                  | The storage path of the model file to be converted. The program structure file and parameter file are saved in this directory.|
    | `model_filename`  | str  | None               | The name of the file storing the model Inference Program structure that needs to be converted. If set to None, use `__model__` as the default filename |
    | `params_filename` | str  | None               | The name of the file that stores all parameters of the model that need to be transformed. It needs to be specified if and only if all model parameters are stored in a single binary file. If the model parameters are stored in separate files, set it to None |
    | `serving_server`  | str  | `"serving_server"` | The storage path of the converted model files and configuration files. Default is serving_server |
    | `serving_client`  | str  | `"serving_client"` | The converted client configuration file storage path. Default is |

- Download and unzip the index of the retrieval library that has been built
    ```shell
    # Go back to the deploy directory
    cd ../
    # Download the built retrieval library index
    wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar
    # Decompress the built retrieval library index
    tar -xf drink_dataset_v1.0.tar
    ```
<a name="4.2"></a>
### 4.2 Service deployment and request

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
  readme.md # Recognition model service deployment documents
  run_cpp_serving.sh # Script to start C++ Pipeline Serving deployment
  test_cpp_serving_client.py # Script for sending C++ Pipeline serving prediction requests by rpc
  ```

<a name="4.2.1"></a>
#### 4.2.1 Python Serving

- Start the service:
  ```shell
  # Start the service and save the running log in log.txt
  python3.7 recognition_web_service.py &>log.txt &
  ```

- send request:
  ```shell
  python3.7 pipeline_http_client.py
  ```
  After a successful run, the results of the model prediction will be printed in the cmd window, and the results are as follows:
  ```log
  {'err_no': 0, 'err_msg': '', 'key': ['result'], 'value': ["[{'bbox': [345, 95, 524, 576], 'rec_docs': '红牛-强化型', 'rec_scores': 0.79903316}]"], 'tensors': []}
  ```

<a name="4.2.2"></a>
#### 4.2.2 C++ Serving

Different from Python Serving, the C++ Serving client calls C++ OP to predict, so before starting the service, you need to compile and install the serving server package, and set `SERVING_BIN`.
- Compile and install the Serving server package
  ```shell
  # Enter the working directory
  cd PaddleClas/deploy/paddleserving
  # One-click compile and install Serving server, set SERVING_BIN
  source ./build_server.sh python3.7
  ```
  **Note:** The path set by [build_server.sh](../../../deploy/paddleserving/build_server.sh#L55-L62) may need to be modified according to the actual machine environment such as CUDA, python version, etc., and then compiled.

- The input and output format used by C++ Serving is different from that of Python, so you need to execute the following command to overwrite the files below [3.1](#31-model-conversion) by copying the 4 files to get the corresponding 4 prototxt files in the folder.
  ```shell
  # Enter PaddleClas/deploy directory
  cd PaddleClas/deploy/

  # Overwrite prototxt file
  \cp ./paddleserving/recognition/preprocess/general_PPLCNet_x2_5_lite_v1.0_serving/*.prototxt ./models/general_PPLCNet_x2_5_lite_v1.0_serving/
  \cp ./paddleserving/recognition/preprocess/general_PPLCNet_x2_5_lite_v1.0_client/*.prototxt ./models/general_PPLCNet_x2_5_lite_v1.0_client/
  \cp ./paddleserving/recognition/preprocess/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/*.prototxt ./models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
  \cp ./paddleserving/recognition/preprocess/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/*.prototxt ./models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/
  ```

- Start the service:
  ```shell
  # Enter the working directory
  cd PaddleClas/deploy/paddleserving/recognition

  # The default port number is 9400; the running log is saved in log_PPShiTu.txt by default
  # CPU deployment
  sh run_cpp_serving.sh
  # GPU deployment, and specify card 0
  sh run_cpp_serving.sh 0
  ```

- send request:
  ```shell
  # send service request
  python3.7 test_cpp_serving_client.py
  ```
  After a successful run, the results of the model predictions are printed in the client's terminal window as follows:
  ```log
  WARNING: Logging before InitGoogleLogging() is written to STDERR
  I0614 03:01:36.273097 6084 naming_service_thread.cpp:202] brpc::policy::ListNamingService("127.0.0.1:9400"): added 1
  I0614 03:01:37.393564 6084 general_model.cpp:490] [client]logid=0,client_cost=1107.82ms,server_cost=1101.75ms.
  [{'bbox': [345, 95, 524, 585], 'rec_docs': '红牛-强化型', 'rec_scores': 0.8073724}]
  ```

- close the service:
  If the service program is running in the foreground, you can press `Ctrl+C` to terminate the server program; if it is running in the background, you can use the kill command to close related processes, or you can execute the following command in the path where the service program is started to terminate the server program:
  ```bash
  python3.7 -m paddle_serving_server.serve stop
  ```
  After the execution is completed, the `Process stopped` message appears, indicating that the service was successfully shut down.


<a name="5"></a>
## 5.FAQ

**Q1**： After sending a request, no result is returned or the output is prompted with a decoding error.

**A1**： Please turn off the proxy before starting the service and sending requests, try the following command:

```
unset https_proxy
unset http_proxy
```

For more types of service deployment, such as `RPC prediction services`, you can refer to the [github official website](https://github.com/PaddlePaddle/Serving/tree/v0.7.0/examples) of Serving.
