# Prediction Framework

## Introduction

Models for Paddle are stored in many different forms, which can be roughly divided into two categories：
1. persistable model（the models saved by fluid.save_persistables）
    The weights are saved in checkpoint, which can be loaded to retrain, one scattered weight file saved by persistable stands for one persistable variable in the model, there is no structure information in these variable, so the weights should be used with the model structure.
    ```
    resnet50-vd-persistable/
    ├── bn2a_branch1_mean
    ├── bn2a_branch1_offset
    ├── bn2a_branch1_scale
    ├── bn2a_branch1_variance
    ├── bn2a_branch2a_mean
    ├── bn2a_branch2a_offset
    ├── bn2a_branch2a_scale
    ├── ...
    └── res5c_branch2c_weights
    ```
2. inference model（the models saved by fluid.io.save_inference_model）
    The model saved by this function cam be used for inference directly, compared with the ones saved by persistable, the model structure will be additionally saved in the model, with the weights, the model with trained weights can be reconstruction. as shown in the following figure, the structure information is saved in `model`
    ```
    resnet50-vd-persistable/
    ├── bn2a_branch1_mean
    ├── bn2a_branch1_offset
    ├── bn2a_branch1_scale
    ├── bn2a_branch1_variance
    ├── bn2a_branch2a_mean
    ├── bn2a_branch2a_offset
    ├── bn2a_branch2a_scale
    ├── ...
    ├── res5c_branch2c_weights
    └── model
    ```
    For convenience, all weight files will be saved into a `params` file when saving the inference model on Paddle, as shown below：
    ```
    resnet50-vd
    ├── model
    └── params
    ```

Both the training engine and the prediction engine in Paddle support the model's e inference, but the back propagation is not performed during the inference, so it can be customized optimization (such as layer fusion, kernel selection, etc.) to achieve low latency and high throughput during inference. The training engine can support either the persistable model or the inference model, and the prediction engine only supports the inference model, so three different inferences are derived：

1. prediction engine + inference model
2. training engine + inference model
3. training engine + inference model

Regardless of the inference method, it basically includes the following main steps：
+ Engine Build
+ Make Data to Be Predicted
+ Perform Predictions
+ Result Analysis

There are two main differences in different inference methods: building the engine and executing the forecast. The following sections will be introduced in detail


## Model Transformation

During training, we usually save some checkpoints (persistable models). These are just model weight files and cannot be directly loaded by the prediction engine to predict, so we usually find suitable checkpoints after the training and convert them to inference model. There are two main steps: 1. Build a training engine, 2. Save the inference model, as shown below.

```python
import fluid

from ppcls.modeling.architectures.resnet_vd import ResNet50_vd

place = fluid.CPUPlace()
exe = fluid.Executor(place)
startup_prog = fluid.Program()
infer_prog = fluid.Program()
with fluid.program_guard(infer_prog, startup_prog):
    with fluid.unique_name.guard():
        image = create_input()
        image = fluid.data(name='image', shape=[None, 3, 224, 224], dtype='float32')
        out = ResNet50_vd.net(input=input, class_dim=1000)

infer_prog = infer_prog.clone(for_test=True)
fluid.load(program=infer_prog, model_path=the path of persistable model, executor=exe)

fluid.io.save_inference_model(
        dirname='./output/',
        feeded_var_names=[image.name],
        main_program=infer_prog,
        target_vars=out,
        executor=exe,
        model_filename='model',
        params_filename='params')
```

A complete example is provided in the `tools/export_model.py`, just execute the following command to complete the conversion：

```python
python tools/export_model.py \
    --m=the name of model \
    --p=the path of persistable model\
    --o=the saved path of model and params
```

## Prediction engine + inference model

The complete example is provided in the `tools/infer/predict.py`，just execute the following command to complete the prediction:

```
python ./tools/infer/predict.py \
    -i=./test.jpeg \
    -m=./resnet50-vd/model \
    -p=./resnet50-vd/params \
    --use_gpu=1 \
    --use_tensorrt=True
```

Parameter Description：
+ `image_file`(shortening i)：the path of images which are needed to predict，such as `./test.jpeg`.
+ `model_file`(shortening m)：the path of weights folder，such as `./resnet50-vd/model`.
+ `params_file`(shortening p)：the path of weights file，such as `./resnet50-vd/params`.
+ `batch_size`(shortening b)：batch size，such as  `1`.
+ `ir_optim` whether to use `IR` optimization, default: True.
+ `use_tensorrt`: whether to use TensorRT prediction engine, default:True.
+ `gpu_mem`： Initial allocation of GPU memory, the unit is M.
+ `use_gpu`: whether to use GPU, default: True.
+ `enable_benchmark`：whether to use benchmark, default: False.
+ `model_name`：the name of model.

NOTE：
when using benchmark, we use tersorrt by default to make predictions on Paddle.


Building prediction engine：

```python
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
config = AnalysisConfig(the path of model file, the path of params file)
config.enable_use_gpu(8000, 0)
config.disable_glog_info()
config.switch_ir_optim(True)
config.enable_tensorrt_engine(
        precision_mode=AnalysisConfig.Precision.Float32,
        max_batch_size=1)

# no zero copy方式需要去除fetch feed op
config.switch_use_feed_fetch_ops(False)

predictor = create_paddle_predictor(config)
```

Prediction Execution：

```python
import numpy as np

input_names = predictor.get_input_names()
input_tensor = predictor.get_input_tensor(input_names[0])
input = np.random.randn(1, 3, 224, 224).astype("float32")
input_tensor.reshape([1, 3, 224, 224])
input_tensor.copy_from_cpu(input)
predictor.zero_copy_run()
```

More parameters information can be refered in [Paddle Python prediction API](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/python_infer_cn.html). If you need to predict in the environment of business, we recommand you to use [Paddel C++ prediction API](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/native_infer.html)，a rich pre-compiled prediction library is provided in the offical website[Paddle C++ prediction library](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。


By default, Paddle's wheel package does not include the TensorRT prediction engine. If you need to use TensorRT for prediction optimization, you need to compile the corresponding wheel package yourself. For the compilation method, please refer to Paddle's compilation guide. [Paddle compilation](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)。

## Training engine + persistable model prediction

A complete example is provided in the `tools/infer/infer.py`, just execute the following command to complete the prediction：

```python
python tools/infer/infer.py \
    --i=the path of images which are needed to predict \
    --m=the name of model \
    --p=the path of persistable model \
    --use_gpu=True
```

Parameter Description：
+ `image_file`(shortening i)：the path of images which are needed to predict，such as `./test.jpeg`
+ `model_file`(shortening m)：the path of weights folder，such as `./resnet50-vd/model`
+ `params_file`(shortening p)：the path of weights file，such as `./resnet50-vd/params`
+ `use_gpu` : whether to use GPU, default: True.


Training Engine Construction：

Since the persistable model does not contain the structural information of the model, it is necessary to construct the network structure first, and then load the weights to build the training engine。

```python
import fluid
from ppcls.modeling.architectures.resnet_vd import ResNet50_vd

place = fluid.CPUPlace()
exe = fluid.Executor(place)
startup_prog = fluid.Program()
infer_prog = fluid.Program()
with fluid.program_guard(infer_prog, startup_prog):
    with fluid.unique_name.guard():
        image = create_input()
        image = fluid.data(name='image', shape=[None, 3, 224, 224], dtype='float32')
        out = ResNet50_vd.net(input=input, class_dim=1000)
infer_prog = infer_prog.clone(for_test=True)
fluid.load(program=infer_prog, model_path=the path of persistable model, executor=exe)
```

Perform inference：

```python
outputs = exe.run(infer_prog,
        feed={image.name: data},
        fetch_list=[out.name],
        return_numpy=False)
```

For the above parameter descriptions, please refer to the official website [fluid.Executor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/executor_cn/Executor_cn.html)

## Training engine + inference model prediction

A complete example is provided in `tools/infer/py_infer.py`, just execute the following command to complete the prediction：

```python
python tools/infer/py_infer.py \
    --i=the path of images \
    --d=the path of saved model \
    --m=the path of saved model file \
    --p=the path of saved weight file \
    --use_gpu=True
```
+ `image_file`(shortening i)：the path of images which are needed to predict，如 `./test.jpeg`
+ `model_file`(shortening m)：the path of model file，如 `./resnet50_vd/model`
+ `params_file`(shortening p)：the path of weights file，如 `./resnet50_vd/params`
+ `model_dir`(shortening d)：the folder of model，如`./resent50_vd`
+ `use_gpu`：whether to use GPU, default: True

Training engine build

Since inference model contains the structure of model, we do not need to construct the model before, load the model file and weights file directly to bulid training engine.

```python
import fluid

place = fluid.CPUPlace()
exe = fluid.Executor(place)
[program, feed_names, fetch_lists] = fluid.io.load_inference_model(
        the path of saved model,
        exe,
        model_filename=the path of model file,
        params_filename=the path of weights file)
compiled_program = fluid.compiler.CompiledProgram(program)
```

> `load_inference_model` Not only supports scattered weight file collection, but also supports a single weight file。

Perform inference：

```python
outputs = exe.run(compiled_program,
        feed={feed_names[0]: data},
        fetch_list=fetch_lists,
        return_numpy=False)
```

For the above parameter descriptions, please refer to the official website [fluid.Executor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/executor_cn/Executor_cn.html)
