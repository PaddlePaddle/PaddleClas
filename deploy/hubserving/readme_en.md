English | [简体中文](readme.md)

# Service deployment based on PaddleHub Serving

PaddleClas supports rapid service deployment through PaddleHub. Currently, the deployment of image classification is supported. Please look forward to the deployment of image recognition.

## Catalogue
- [1 Introduction](#1-introduction)
- [2. Prepare the environment](#2-prepare-the-environment)
- [3. Download the inference model](#3-download-the-inference-model)
- [4. Install the service module](#4-install-the-service-module)
- [5. Start service](#5-start-service)
  - [5.1 Start with command line parameters](#51-start-with-command-line-parameters)
  - [5.2 Start with configuration file](#52-start-with-configuration-file)
- [6. Send prediction requests](#6-send-prediction-requests)
- [7. User defined service module modification](#7-user-defined-service-module-modification)


<a name="1"></a>
## 1 Introduction

The hubserving service deployment configuration service package `clas` contains 3 required files, the directories are as follows:

```shell
deploy/hubserving/clas/
├── __init__.py # Empty file, required
├── config.json # Configuration file, optional, passed in as a parameter when starting the service with configuration
├── module.py # The main module, required, contains the complete logic of the service
└── params.py # Parameter file, required, including model path, pre- and post-processing parameters and other parameters
```


<a name="2"></a>
## 2. Prepare the environment
```shell
# Install paddlehub, version 2.1.0 is recommended
python3.7 -m pip install paddlehub==2.1.0 --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```


<a name="3"></a>
## 3. Download the inference model

Before installing the service module, you need to prepare the inference model and put it in the correct path. The default model path is:

* Classification inference model structure file: `PaddleClas/inference/inference.pdmodel`
* Classification inference model weight file: `PaddleClas/inference/inference.pdiparams`

**Notice**:
* Model file paths can be viewed and modified in `PaddleClas/deploy/hubserving/clas/params.py`:

  ```python
  "inference_model_dir": "../inference/"
  ```
* Model files (including `.pdmodel` and `.pdiparams`) must be named `inference`.
* We provide a large number of pre-trained models based on the ImageNet-1k dataset. For the model list and download address, see [Model Library Overview](../../docs/en/algorithm_introduction/ImageNet_models_en.md), or you can use your own trained and converted models.


<a name="4"></a>
## 4. Install the service module

* In the Linux environment, the installation example is as follows:
  ```shell
  cd PaddleClas/deploy
  # Install the service module:
  hub install hubserving/clas/
  ```

* In the Windows environment (the folder separator is `\`), the installation example is as follows:

  ```shell
  cd PaddleClas\deploy
  # Install the service module:
  hub install hubserving\clas\
  ```


<a name="5"></a>
## 5. Start service


<a name="5.1"></a>
### 5.1 Start with command line parameters

This method only supports prediction using CPU. Start command:

```shell
hub serving start \
--modules clas_system
--port 8866
```
This completes the deployment of a serviced API, using the default port number 8866.

**Parameter Description**:
| parameters         | uses                |
| ------------------ | ------------------- |
| --modules/-m       | [**required**] PaddleHub Serving pre-installed model, listed in the form of multiple Module==Version key-value pairs<br>*`When no Version is specified, the latest is selected by default version`*                       |
| --port/-p          | [**OPTIONAL**] Service port, default is 8866                                                                                                                                                                               |
| --use_multiprocess | [**Optional**] Whether to enable the concurrent mode, the default is single-process mode, it is recommended to use this mode for multi-core CPU machines<br>*`Windows operating system only supports single-process mode`* |
| --workers          | [**Optional**] The number of concurrent tasks specified in concurrent mode, the default is `2*cpu_count-1`, where `cpu_count` is the number of CPU cores                                                                   |
For more deployment details, see [PaddleHub Serving Model One-Click Service Deployment](https://paddlehub.readthedocs.io/zh_CN/release-v2.1/tutorial/serving.html)

<a name="5.2"></a>
### 5.2 Start with configuration file

This method only supports prediction using CPU or GPU. Start command:

```shell
hub serving start -c config.json
```

Among them, the format of `config.json` is as follows:

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

**Parameter Description**:
* The configurable parameters in `init_args` are consistent with the `_initialize` function interface in `module.py`. in,
  - When `use_gpu` is `true`, it means to use GPU to start the service.
  - When `enable_mkldnn` is `true`, it means to use MKL-DNN acceleration.
* The configurable parameters in `predict_args` are consistent with the `predict` function interface in `module.py`.

**Notice**:
* When using the configuration file to start the service, the parameter settings in the configuration file will be used, and other command line parameters will be ignored;
* If you use GPU prediction (ie, `use_gpu` is set to `true`), you need to set the `CUDA_VISIBLE_DEVICES` environment variable to specify the GPU card number used before starting the service, such as: `export CUDA_VISIBLE_DEVICES=0`;
* **`use_gpu` cannot be `true`** at the same time as `use_multiprocess`;
* ** When both `use_gpu` and `enable_mkldnn` are `true`, `enable_mkldnn` will be ignored and GPU** will be used.

If you use GPU No. 3 card to start the service:

```shell
cd PaddleClas/deploy
export CUDA_VISIBLE_DEVICES=3
hub serving start -c hubserving/clas/config.json
```

<a name="6"></a>
## 6. Send prediction requests

After configuring the server, you can use the following command to send a prediction request to get the prediction result:

```shell
cd PaddleClas/deploy
python3.7 hubserving/test_hubserving.py \
--server_url http://127.0.0.1:8866/predict/clas_system \
--image_file ./hubserving/ILSVRC2012_val_00006666.JPEG \
--batch_size 8
```
**Predicted output**
```log
The result(s): class_ids: [57, 67, 68, 58, 65], label_names: ['garter snake, grass snake', 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'sidewinder, horned rattlesnake, Crotalus cerastes' , 'water snake', 'sea snake'], scores: [0.21915, 0.15631, 0.14794, 0.13177, 0.12285]
The average time of prediction cost: 2.970 s/image
The average time cost: 3.014 s/image
The average top-1 score: 0.110
```

**Script parameter description**:
* **server_url**: Service address, the format is `http://[ip_address]:[port]/predict/[module_name]`.
* **image_path**: The test image path, which can be a single image path or an image collection directory path.
* **batch_size**: [**OPTIONAL**] Make predictions in `batch_size` size, default is `1`.
* **resize_short**: [**optional**] When preprocessing, resize by short edge, default is `256`.
* **crop_size**: [**Optional**] The size of the center crop during preprocessing, the default is `224`.
* **normalize**: [**Optional**] Whether to perform `normalize` during preprocessing, the default is `True`.
* **to_chw**: [**Optional**] Whether to adjust to `CHW` order when preprocessing, the default is `True`.

**Note**: If you use `Transformer` series models, such as `DeiT_***_384`, `ViT_***_384`, etc., please pay attention to the input data size of the model, you need to specify `--resize_short=384 -- crop_size=384`.

**Return result format description**:
The returned result is a list (list), including the top-k classification results, the corresponding scores, and the time-consuming prediction of this image, as follows:
```shell
list: return result
└──list: first image result
   ├── list: the top k classification results, sorted in descending order of score
   ├── list: the scores corresponding to the first k classification results, sorted in descending order of score
   └── float: The image classification time, in seconds
```



<a name="7"></a>
## 7. User defined service module modification

If you need to modify the service logic, you need to do the following:

1. Stop the service
    ```shell
    hub serving stop --port/-p XXXX
    ```

2. Go to the corresponding `module.py` and `params.py` and other files to modify the code according to actual needs. `module.py` needs to be reinstalled after modification (`hub install hubserving/clas/`) and deployed. Before deploying, you can use the `python3.7 hubserving/clas/module.py` command to quickly test the code ready for deployment.

3. Uninstall the old service pack
    ```shell
    hub uninstall clas_system
     ```

4. Install the new modified service pack
     ```shell
     hub install hubserving/clas/
     ```

5. Restart the service
     ```shell
     hub serving start -m clas_system
     ```

**Notice**:
Common parameters can be modified in `PaddleClas/deploy/hubserving/clas/params.py`:
   * To replace the model, you need to modify the model file path parameters:
     ```python
     "inference_model_dir":
     ```
   * Change the number of `top-k` results returned when postprocessing:
     ```python
     'topk':
     ```
   * The mapping file corresponding to the lable and class id when changing the post-processing:
     ```python
     'class_id_map_file':
     ```

In order to avoid unnecessary delay and be able to predict with batch_size, data preprocessing logic (including `resize`, `crop` and other operations) is completed on the client side, so it needs to modify data preprocessing logic related code in [PaddleClas/deploy/hubserving/test_hubserving.py# L41-L47](./test_hubserving.py#L41-L47) and [PaddleClas/deploy/hubserving/test_hubserving.py#L51-L76](./test_hubserving.py#L51-L76).
