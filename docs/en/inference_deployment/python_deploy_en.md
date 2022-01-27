# Infering based on Python prediction engine

The inference model (the model saved by `paddle.jit.save`) is generally a solidified model saved after the model training is completed, and is mostly used to give prediction in deployment.

The model saved during the training process is the checkpoints model, which saves the parameters of the model and is mostly used to resume training.

Compared with the checkpoints model, the inference model will additionally save the structural information of the model. Therefore, it is easier to deploy because the model structure and model parameters are already solidified in the inference model file, and is suitable for integration with actual systems.

Please refer to the document [install paddle](../installation/install_paddle_en.md) and [install paddleclas](../installation/install_paddleclas_en.md) to prepare the environment.

---

## Catalogue

- [1. Image classification inference](#1)
- [2. Mainbody detection model inference](#2)
- [3. Feature Extraction model inference](#3)
- [4. Concatenation of mainbody detection, feature extraction and vector search](#4)


<a name="1"></a>
## 1. Image classification inference

First, please refer to the document [export model](./export_model_en.md) to prepare the inference model files. All the command should be run under `deploy` folder of PaddleClas:

```shell
cd deploy
```

For classification model inference, you can execute the following commands:

```shell
python python/predict_cls.py -c configs/inference_cls.yaml
```

In the configuration file `configs/inference_cls.yaml`, the following fields are used to configure prediction parameters:
* `Global.infer_imgs`: The path of image to be predicted;
* `Global.inference_model_dir`: The directory of inference model files. There should be contain the model files (`inference.pdmodel` and `inference.pdiparams`);
* `Global.use_tensorrt`: Whether use `TensorRT`, `False` by default;
* `Global.use_gpu`: Whether use GPU, `True` by default;
* `Global.enable_mkldnn`: Whether use `MKL-DNN`, `False` by default. Valid only when `use_gpu` is `False`;
* `Global.use_fp16`: Whether use `FP16`, `False` by default;
* `PreProcess`: To config the preprocessing of image to be predicted;
* `PostProcess`: To config the postprocessing of prediction results;
* `PostProcess.Topk.class_id_map_file`: The path of file mapping label and class id. By default ImageNet1k (`./utils/imagenet1k_label_list.txt`).

**Notice**:
* If VisionTransformer series models used, such as `DeiT_***_384`, `ViT_***_384`, please notice the size of model input. And you could need to specify the `PreProcess.resize_short=384`, `PreProcess.resize=384`.
* If you want to improve the speed of the evaluation, it is recommended to enable TensorRT when using GPU, and MKL-DNN when using CPU.

```shell
python python/predict_cls.py -c configs/inference_cls.yaml -o Global.infer_imgs=images/ILSVRC2012_val_00010010.jpeg
```

If you want to use the CPU for prediction, you can switch value of `use_gpu` in config file to `False`. Or you can execute the command as follows
```
python python/predict_cls.py -c configs/inference_cls.yaml  -o Global.use_gpu=False
```

<a name="2"></a>
## 2. Mainbody detection model inference

The following will introduce the mainbody detection model inference. All the command should be run under `deploy` folder of PaddleClas:

```shell
cd deploy
```

For mainbody detection model inference, you can execute the following commands:

```shell
mkdir -p models
cd models
# download mainbody detection inference model
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar && tar -xf ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
cd ..
# predict
python python/predict_det.py -c configs/inference_det.yaml
```

The input example image is as follows:
[](../images/recognition/product_demo/wangzai.jpg)

The output will be:

```text
[{'class_id': 0, 'score': 0.4762245, 'bbox': array([305.55115, 226.05322, 776.61084, 930.42395], dtype=float32), 'label_name': 'foreground'}]
```

And the visualise result is as follows:
[](../images/recognition/product_demo/wangzai_det_result.jpg)

If you want to detect another image, you can change the value of `infer_imgs` in `configs/inference_det.yaml`,
or you can use `-o Global.infer_imgs` argument. For example, if you want to detect `images/anmuxi.jpg`:

```shell
python python/predict_det.py -c configs/inference_det.yaml -o Global.infer_imgs=images/anmuxi.jpg
```

If you want to use the CPU for prediction, you can switch value of `use_gpu` in config file to `False`. Or you can execute the command as follows
```
python python/predict_det.py -c configs/inference_det.yaml  -o Global.use_gpu=False
```

<a name="3"></a>
## 3. Feature Extraction model inference

First, please refer to the document [export model](./export_model_en.md) to prepare the inference model files. All the command should be run under `deploy` folder of PaddleClas:

```shell
cd deploy
```

For feature extraction model inference, you can execute the following commands:

```shell
mkdir -p models
cd models
# download feature extraction inference model
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/product_ResNet50_vd_aliproduct_v1.0_infer.tar && tar -xf product_ResNet50_vd_aliproduct_v1.0_infer.tar
cd ..
# predict
python python/predict_rec.py -c configs/inference_rec.yaml
```
You can get a 512-dim feature printed in the command line.

If you want to extract feature of another image, you can change the value of `infer_imgs` in `configs/inference_rec.yaml`,
or you can use `-o Global.infer_imgs` argument. For example, if you want to try `images/anmuxi.jpg`:

```shell
python python/predict_rec.py -c configs/inference_rec.yaml -o Global.infer_imgs=images/anmuxi.jpg
```

If you want to use the CPU for prediction, you can switch value of `use_gpu` in config file to `False`. Or you can execute the command as follows

```
python python/predict_rec.py -c configs/inference_rec.yaml  -o Global.use_gpu=False
```

<a name="4"></a>
## 4. Concatenation of mainbody detection, feature extraction and vector search
 Please refer to [Quick Start of Recognition](../quick_start/quick_start_recognition_en.md)
