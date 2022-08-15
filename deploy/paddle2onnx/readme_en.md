# Paddle2ONNX: Converting To ONNX and Deployment

This section introduce that how to convert the Paddle Inference Model ResNet50_vd to ONNX model and deployment based on ONNX engine.

## 1. Installation

First, you need to install Paddle2ONNX and onnxruntime. Paddle2ONNX is a toolkit to convert Paddle Inference Model to ONNX model. Please refer to [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_en.md) for more information.

- Paddle2ONNX Installation
```
python3.7 -m pip install paddle2onnx
```

- ONNX Installation
```
python3.7 -m pip install onnxruntime
```

## 2. Converting to ONNX

Download the Paddle Inference Model ResNet50_vd:

```
cd deploy
mkdir models && cd models
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar && tar xf ResNet50_vd_infer.tar
cd ..
```

Converting to ONNX model:

```
paddle2onnx --model_dir=./models/ResNet50_vd_infer/ \
--model_filename=inference.pdmodel \
--params_filename=inference.pdiparams \
--save_file=./models/ResNet50_vd_infer/inference.onnx \
--opset_version=10 \
--enable_onnx_checker=True
```

After running the above command, the ONNX model file converted would be save in  `./models/ResNet50_vd_infer/`.

## 3. Deployment

Deployment with ONNX model, command is as shown below.

```
python3.7 python/predict_cls.py \
-c configs/inference_cls.yaml \
-o Global.use_onnx=True \
-o Global.use_gpu=False \
-o Global.inference_model_dir=./models/ResNet50_vd_infer
```

The prediction results:

```
ILSVRC2012_val_00000010.jpeg:   class id(s): [153, 204, 229, 332, 155], score(s): [0.69, 0.10, 0.02, 0.01, 0.01], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'Lhasa, Lhasa apso', 'Old English sheepdog, bobtail', 'Angora, Angora rabbit', 'Shih-Tzu']
```
