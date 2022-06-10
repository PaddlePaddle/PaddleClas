# paddle2onnx 模型转化与预测

## 目录

- [paddle2onnx 模型转化与预测](#paddle2onnx-模型转化与预测)
  - [1. 环境准备](#1-环境准备)
  - [2. 模型转换](#2-模型转换)
  - [3. onnx 预测](#3-onnx-预测)

## 1. 环境准备

需要准备 Paddle2ONNX 模型转化环境，和 ONNX 模型预测环境。

Paddle2ONNX 支持将 PaddlePaddle inference 模型格式转化到 ONNX 模型格式，算子目前稳定支持导出 ONNX Opset 9~11。
更多细节可参考 [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX#paddle2onnx)

- 安装 Paddle2ONNX
    ```shell
    python3.7 -m pip install paddle2onnx
    ```

- 安装 ONNX 推理引擎
    ```shell
    python3.7 -m pip install onnxruntime
    ```
下面以 ResNet50_vd 为例，介绍如何将 PaddlePaddle inference 模型转换为 ONNX 模型，并基于 ONNX 引擎预测。

## 2. 模型转换

- ResNet50_vd inference模型下载

    ```shell
    cd deploy
    mkdir models && cd models
    wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar && tar xf ResNet50_vd_infer.tar
    cd ..
    ```

- 模型转换

    使用 Paddle2ONNX 将 Paddle 静态图模型转换为 ONNX 模型格式：
    ```shell
    paddle2onnx --model_dir=./models/ResNet50_vd_infer/ \
    --model_filename=inference.pdmodel \
    --params_filename=inference.pdiparams \
    --save_file=./models/ResNet50_vd_infer/inference.onnx \
    --opset_version=10 \
    --enable_onnx_checker=True
    ```

转换完毕后，生成的ONNX 模型 `inference.onnx` 会被保存在 `./models/ResNet50_vd_infer/` 路径下

## 3. onnx 预测

执行如下命令：
```shell
python3.7 python/predict_cls.py \
-c configs/inference_cls.yaml \
-o Global.use_onnx=True \
-o Global.use_gpu=False \
-o Global.inference_model_dir=./models/ResNet50_vd_infer
```

结果如下：
```
ILSVRC2012_val_00000010.jpeg:   class id(s): [153, 204, 229, 332, 155], score(s): [0.69, 0.10, 0.02, 0.01, 0.01], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'Lhasa, Lhasa apso', 'Old English sheepdog, bobtail', 'Angora, Angora rabbit', 'Shih-Tzu']
```
