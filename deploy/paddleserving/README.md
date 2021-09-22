# PaddleClas Pipeline WebService

(English|[简体中文](./README_CN.md))

PaddleClas provides two service deployment methods:
- Based on **PaddleHub Serving**: Code path is "`./deploy/hubserving`". Please refer to the [tutorial](../../deploy/hubserving/readme_en.md)
- Based on **PaddleServing**: Code path is "`./deploy/paddleserving`".  if you prefer retrieval_based image reocognition service, please refer to [tutorial](./recognition/README.md)，if you'd like image classification service, Please follow this tutorial.

# Image Classification Service deployment based on PaddleServing  

This document will introduce how to use the [PaddleServing](https://github.com/PaddlePaddle/Serving/blob/develop/README.md) to deploy the ResNet50_vd model as a pipeline online service.

Some Key Features of Paddle Serving:
- Integrate with Paddle training pipeline seamlessly, most paddle models can be deployed with one line command.
- Industrial serving features supported, such as models management, online loading, online A/B testing etc.
- Highly concurrent and efficient communication between clients and servers supported.

The introduction and tutorial of Paddle Serving service deployment framework reference [document](https://github.com/PaddlePaddle/Serving/blob/develop/README.md).


## Contents
- [Environmental preparation](#environmental-preparation)
- [Model conversion](#model-conversion)
- [Paddle Serving pipeline deployment](#paddle-serving-pipeline-deployment)
- [FAQ](#faq)

<a name="environmental-preparation"></a>
## Environmental preparation

PaddleClas operating environment and PaddleServing operating environment are needed.

1. Please prepare PaddleClas operating environment reference [link](../../docs/zh_CN/tutorials/install.md).
   Download the corresponding paddle whl package according to the environment, it is recommended to install version 2.1.0.

2. The steps of PaddleServing operating environment prepare are as follows:

    Install serving which used to start the service
    ```
    pip3 install paddle-serving-server==0.6.1 # for CPU
    pip3 install paddle-serving-server-gpu==0.6.1 # for GPU
    # Other GPU environments need to confirm the environment and then choose to execute the following commands
    pip3 install paddle-serving-server-gpu==0.6.1.post101 # GPU with CUDA10.1 + TensorRT6
    pip3 install paddle-serving-server-gpu==0.6.1.post11 # GPU with CUDA11 + TensorRT7
    ```

3. Install the client to send requests to the service
    In [download link](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md) find the client installation package corresponding to the python version.
    The python3.7 version is recommended here:

    ```
    wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_client-0.0.0-cp37-none-any.whl
    pip3 install paddle_serving_client-0.0.0-cp37-none-any.whl
    ```

4. Install serving-app
    ```
    pip3 install paddle-serving-app==0.6.1
    ```

   **note:** If you want to install the latest version of PaddleServing, refer to [link](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md).


<a name="model-conversion"></a>
## Model conversion
When using PaddleServing for service deployment, you need to convert the saved inference model into a serving model that is easy to deploy.

Firstly, download the inference model of ResNet50_vd
```
# Download and unzip the ResNet50_vd model
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar  && tar xf ResNet50_vd_infer.tar
```

Then, you can use installed paddle_serving_client tool to convert inference model to mobile model.
```
#  ResNet50_vd model conversion
python3 -m paddle_serving_client.convert --dirname ./ResNet50_vd_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./ResNet50_vd_serving/ \
                                         --serving_client ./ResNet50_vd_client/
```

After the ResNet50_vd inference model is converted, there will be additional folders of `ResNet50_vd_serving` and `ResNet50_vd_client` in the current folder, with the following format:
```
|- ResNet50_vd_client/
  |- __model__  
  |- __params__
  |- serving_server_conf.prototxt  
  |- serving_server_conf.stream.prototxt

|- ResNet50_vd_client
  |- serving_client_conf.prototxt  
  |- serving_client_conf.stream.prototxt
```

Once you have the model file for deployment, you need to change the alias name in `serving_server_conf.prototxt`: Change `alias_name` in `feed_var` to `image`, change `alias_name` in `fetch_var` to `prediction`,
The modified serving_server_conf.prototxt file is as follows:
```
feed_var {
  name: "inputs"
  alias_name: "image"
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

<a name="paddle-serving-pipeline-deployment"></a>
## Paddle Serving pipeline deployment

1. Download the PaddleClas code, if you have already downloaded it, you can skip this step.
    ```
    git clone https://github.com/PaddlePaddle/PaddleClas

    # Enter the working directory  
    cd PaddleClas/deploy/paddleserving/
    ```

    The paddleserving directory contains the code to start the pipeline service and send prediction requests, including:
    ```
    __init__.py
    config.yml                # configuration file of starting the service
    pipeline_http_client.py   # script to send pipeline prediction request by http
    pipeline_rpc_client.py    # script to send pipeline prediction request by rpc
    classification_web_service.py   # start the script of the pipeline server
    ```

2. Run the following command to start the service.
    ```
    # Start the service and save the running log in log.txt
    python3 classification_web_service.py &>log.txt &
    ```
    After the service is successfully started, a log similar to the following will be printed in log.txt
    ![](./imgs/start_server.png)

3. Send service request
    ```
    python3 pipeline_http_client.py
    ```
    After successfully running, the predicted result of the model will be printed in the cmd window. An example of the result is:
    ![](./imgs/results.png)

    Adjust the number of concurrency in config.yml to get the largest QPS. 

    ```
    op:
        concurrency: 8
        ...
    ```

    Multiple service requests can be sent at the same time if necessary.

    The predicted performance data will be automatically written into the `PipelineServingLogs/pipeline.tracer` file.

<a name="faq"></a>
## FAQ
**Q1**: No result return after sending the request.

**A1**: Do not set the proxy when starting the service and sending the request. You can close the proxy before starting the service and before sending the request. The command to close the proxy is:
```
unset https_proxy
unset http_proxy
```  
