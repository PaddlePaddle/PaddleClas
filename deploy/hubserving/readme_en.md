English | [简体中文](readme.md)

# Service deployment based on PaddleHub Serving  

HubServing service pack contains 3 files, the directory is as follows:  
```
deploy/hubserving/clas/
  └─  __init__.py    Empty file, required
  └─  config.json    Configuration file, optional, passed in as a parameter when using configuration to start the service
  └─  module.py      Main module file, required, contains the complete logic of the service
  └─  params.py      Parameter file, required, including parameters such as model path, pre- and post-processing parameters
```

## Quick start service
### 1. Prepare the environment
```shell
# Install paddlehub  
pip3 install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Download inference model
Before installing the service module, you need to prepare the inference model and put it in the correct path. The default model path is:  

```
Model structure file: ./inference/cls_infer.pdmodel
Model parameters file: ./inference/cls_infer.pdiparams
```

**The model path can be found and modified in `params.py`.** More models provided by PaddleClas can be obtained from the [model library](../../docs/en/models/models_intro_en.md). You can also use models trained by yourself.

### 3. Install Service Module

* On Linux platform, the examples are as follows.
```shell
hub install deploy/hubserving/clas/
```

* On Windows platform, the examples are as follows.
```shell
hub install deploy\hubserving\clas\
```

### 4. Start service
#### Way 1. Start with command line parameters (CPU only)

**start command：**  
```shell
$ hub serving start --modules Module1==Version1 \
                    --port XXXX \
                    --use_multiprocess \
                    --workers \
```  
**parameters：**  

|parameters|usage|  
|-|-|  
|--modules/-m|PaddleHub Serving pre-installed model, listed in the form of multiple Module==Version key-value pairs<br>*`When Version is not specified, the latest version is selected by default`*|
|--port/-p|Service port, default is 8866|  
|--use_multiprocess|Enable concurrent mode, the default is single-process mode, this mode is recommended for multi-core CPU machines<br>*`Windows operating system only supports single-process mode`*|
|--workers|The number of concurrent tasks specified in concurrent mode, the default is `2*cpu_count-1`, where `cpu_count` is the number of CPU cores|  

For example, start the 2-stage series service:  
```shell
hub serving start -m clas_system
```  

This completes the deployment of a service API, using the default port number 8866.  

#### Way 2. Start with configuration file（CPU、GPU）
**start command：**  
```shell
hub serving start --config/-c config.json
```  
Wherein, the format of `config.json` is as follows:
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
- The configurable parameters in `init_args` are consistent with the `_initialize` function interface in `module.py`. Among them,
  - when `use_gpu` is `true`, it means that the GPU is used to start the service.
  - when `enable_mkldnn` is `true`, it means that use MKL-DNN to accelerate.
- The configurable parameters in `predict_args` are consistent with the `predict` function interface in `module.py`.

**Note:**  
- When using the configuration file to start the service, other parameters will be ignored.
- If you use GPU prediction (that is, `use_gpu` is set to `true`), you need to set the environment variable CUDA_VISIBLE_DEVICES before starting the service, such as: ```export CUDA_VISIBLE_DEVICES=0```, otherwise you do not need to set it.
- **`use_gpu` and `use_multiprocess` cannot be `true` at the same time.**  
- **When both `use_gpu` and `enable_mkldnn` are set to `true` at the same time, GPU is used to run and `enable_mkldnn` will be ignored.**

For example, use GPU card No. 3 to start the 2-stage series service:
```shell
export CUDA_VISIBLE_DEVICES=3
hub serving start -c deploy/hubserving/clas/config.json
```  

## Send prediction requests
After the service starts, you can use the following command to send a prediction request to obtain the prediction result:  
```shell
python tools/test_hubserving.py server_url image_path
```  

Two parameters need to be passed to the script:
- **server_url**：service address，format of which is
`http://[ip_address]:[port]/predict/[module_name]`  
- **image_path**：Test image path, can be a single image path or an image directory path
- **top_k**：[**Optional**] Return the top `top_k` 's scores ,default by `1`.

**Eg.**
```shell
python tools/test_hubserving.py http://127.0.0.1:8866/predict/clas_system ./deploy/hubserving/ILSVRC2012_val_00006666.JPEG 5
```

## Returned result format
The returned result is a list, including classification results(`clas`), and the `top_k`'s scores(`socres`). And `scores` is a list, consist of `score`.

**Note：** If you need to add, delete or modify the returned fields, you can modify the file `module.py` of the corresponding module. For the complete process, refer to the user-defined modification service module in the next section.

## User defined service module modification
If you need to modify the service logic, the following steps are generally required:

- 1. Stop service
```shell
hub serving stop --port/-p XXXX
```
- 2. Modify the code in the corresponding files, like `module.py` and `params.py`, according to the actual needs.  
For example, if you need to replace the model used by the deployed service, you need to modify model path parameters `cfg.model_file` and `cfg.params_file` in `params.py`. Of course, other related parameters may need to be modified at the same time. Please modify and debug according to the actual situation. It is suggested to run `module.py` directly for debugging after modification before starting the service test.  
- 3. Uninstall old service module
```shell
hub uninstall clas_system
```
- 4. Install modified service module
```shell
hub install deploy/hubserving/clas/
```
- 5. Restart service
```shell
hub serving start -m clas_system
```
