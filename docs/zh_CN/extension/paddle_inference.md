# 分类预测框架

## 一、简介

Paddle 的模型保存有多种不同的形式，大体可分为两类：
1. persistable 模型（fluid.save_persistabels保存的模型）
    一般做为模型的 checkpoint，可以加载后重新训练。persistable 模型保存的是零散的权重文件，每个文件代表模型中的一个 Variable，这些零散的文件不包含结构信息，需要结合模型的结构一起使用。
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
2. inference 模型（fluid.io.save_inference_model保存的模型）
    一般是模型训练完成后保存的固化模型，用于预测部署。与 persistable 模型相比，inference 模型会额外保存模型的结构信息，用于配合权重文件构成完整的模型。如下所示，`model` 中保存的即为模型的结构信息。
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
    为了方便起见，paddle 在保存 inference 模型的时候也可以将所有的权重文件保存成一个`params`文件，如下所示：
    ```
    resnet50-vd
    ├── model
    └── params
    ```

在 Paddle 中训练引擎和预测引擎都支持模型的预测推理，只不过预测引擎不需要进行反向操作，因此可以进行定制型的优化（如层融合，kernel 选择等），达到低时延、高吞吐的目的。训练引擎既可以支持 persistable 模型，也可以支持 inference 模型，而预测引擎只支持 inference 模型，因此也就衍生出了三种不同的预测方式：

1. 预测引擎 + inference 模型
2. 训练引擎 + persistable 模型
3. 训练引擎 + inference 模型

不管是何种预测方式，基本都包含以下几个主要的步骤：
+ 构建引擎
+ 构建待预测数据
+ 执行预测
+ 预测结果解析

不同预测方式，主要有两方面不同：构建引擎和执行预测，以下的几个部分我们会具体介绍。


## 二、模型转换

在任务的训练阶段，通常我们会保存一些 checkpoint（persistable 模型），这些只是模型权重文件，不能直接被预测引擎直接加载预测，所以我们通常会在训练完之后，找到合适的 checkpoint 并将其转换为 inference 模型。主要分为两个步骤：1. 构建训练引擎，2. 保存 inference 模型，如下所示：

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
fluid.load(program=infer_prog, model_path=persistable 模型路径, executor=exe)

fluid.io.save_inference_model(
        dirname='./output/',
        feeded_var_names=[image.name],
        main_program=infer_prog,
        target_vars=out,
        executor=exe,
        model_filename='model',
        params_filename='params')
```

在模型库的 `tools/export_model.py` 中提供了完整的示例，只需执行下述命令即可完成转换：

```python
python tools/export_model.py \
    --m=模型名称 \
    --p=persistable 模型路径 \
    --o=model和params保存路径
```

## 三、预测引擎 + inference 模型预测

在模型库的 `tools/infer/predict.py` 中提供了完整的示例，只需执行下述命令即可完成预测：

```
python ./tools/infer/predict.py \
    -i=./test.jpeg \
    -m=./resnet50-vd/model \
    -p=./resnet50-vd/params \
    --use_gpu=1 \
    --use_tensorrt=True
```

参数说明：
+ `image_file`(简写 i)：待预测的图片文件路径，如 `./test.jpeg`
+ `model_file`(简写 m)：模型文件路径，如 `./resnet50-vd/model`
+ `params_file`(简写 p)：权重文件路径，如 `./resnet50-vd/params`
+ `batch_size`(简写 b)：批大小，如 `1`
+ `ir_optim`：是否使用 `IR` 优化，默认值：True
+ `use_tensorrt`：是否使用 TesorRT 预测引擎，默认值：True
+ `gpu_mem`： 初始分配GPU显存，以M单位
+ `use_gpu`：是否使用 GPU 预测，默认值：True
+ `enable_benchmark`：是否启用benchmark，默认值：False
+ `model_name`：模型名字

注意：
当启用benchmark时，默认开启tersorrt进行预测


构建预测引擎：

```python
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
config = AnalysisConfig(model文件路径, params文件路径)
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

执行预测：

```python
import numpy as np

input_names = predictor.get_input_names()
input_tensor = predictor.get_input_tensor(input_names[0])
input = np.random.randn(1, 3, 224, 224).astype("float32")
input_tensor.reshape([1, 3, 224, 224])
input_tensor.copy_from_cpu(input)
predictor.zero_copy_run()
```

更多预测参数说明可以参考官网 [Paddle Python 预测 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/python_infer_cn.html)。如果需要在业务的生产环境部署，也推荐使用 [Paddel C++ 预测 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/native_infer.html)，官网提供了丰富的预编译预测库 [Paddle C++ 预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。


默认情况下，Paddle 的 wheel 包中是不包含 TensorRT 预测引擎的，如果需要使用 TensorRT 进行预测优化，需要自己编译对应的 wheel 包，编译方式可以参考 Paddle 的编译指南 [Paddle 编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)。

## 四、训练引擎 + persistable 模型预测

在模型库的 `tools/infer/infer.py` 中提供了完整的示例，只需执行下述命令即可完成预测：

```python
python tools/infer/infer.py \
    --i=待预测的图片文件路径 \
    --m=模型名称 \
    --p=persistable 模型路径 \
    --use_gpu=True
```

参数说明：
+ `image_file`(简写 i)：待预测的图片文件路径，如 `./test.jpeg`
+ `model_file`(简写 m)：模型文件路径，如 `./resnet50-vd/model`
+ `params_file`(简写 p)：权重文件路径，如 `./resnet50-vd/params`
+ `use_gpu` : 是否开启GPU训练，默认值：True


训练引擎构建：

由于 persistable 模型不包含模型的结构信息，因此需要先构建出网络结构，然后 load 权重来构建训练引擎。

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
fluid.load(program=infer_prog, model_path=persistable 模型路径, executor=exe)
```

执行预测：

```python
outputs = exe.run(infer_prog,
        feed={image.name: data},
        fetch_list=[out.name],
        return_numpy=False)
```

上述执行预测时候的参数说明可以参考官网 [fluid.Executor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/executor_cn/Executor_cn.html)

## 五、训练引擎 + inference 模型预测

在模型库的 `tools/infer/py_infer.py` 中提供了完整的示例，只需执行下述命令即可完成预测：

```python
python tools/infer/py_infer.py \
    --i=图片路径 \
    --d=模型的存储路径 \
    --m=保存的模型文件 \
    --p=保存的参数文件 \
    --use_gpu=True
```
+ `image_file`(简写 i)：待预测的图片文件路径，如 `./test.jpeg`
+ `model_file`(简写 m)：模型文件路径，如 `./resnet50_vd/model`
+ `params_file`(简写 p)：权重文件路径，如 `./resnet50_vd/params`
+ `model_dir`(简写d)：模型路径，如`./resent50_vd`
+ `use_gpu`：是否开启GPU，默认值：True

训练引擎构建：

由于 inference 模型已包含模型的结构信息，因此不再需要提前构建模型结构，直接 load 模型结构和权重文件来构建训练引擎。

```python
import fluid

place = fluid.CPUPlace()
exe = fluid.Executor(place)
[program, feed_names, fetch_lists] = fluid.io.load_inference_model(
        模型的存储路径,
        exe,
        model_filename=保存的模型文件,
        params_filename=保存的参数文件)
compiled_program = fluid.compiler.CompiledProgram(program)
```

> `load_inference_model` 既支持零散的权重文件集合，也支持融合后的单个权重文件。

执行预测：

```python
outputs = exe.run(compiled_program,
        feed={feed_names[0]: data},
        fetch_list=fetch_lists,
        return_numpy=False)
```

上述执行预测时候的参数说明可以参考官网 [fluid.Executor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/executor_cn/Executor_cn.html)
