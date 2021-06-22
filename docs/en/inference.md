# INFERING BASED ON PYTHON PREDICTION ENGINE

The inference model (the model saved by `paddle.jit.save`) is generally a solidified model saved after the model training is completed, and is mostly used to give prediction in deployment.

The model saved during the training process is the checkpoints model, which saves the parameters of the model and is mostly used to resume training.

Compared with the checkpoints model, the inference model will additionally save the structural information of the model. Therefore, it is easier to deploy because the model structure and model parameters are already solidified in the inference model file, and is suitable for integration with actual systems.

Next, we first introduce how to convert a trained model into an inference model, and then we will introduce mainbody detection, feature extraction based on inference model, 
then we introduce a recognition pipeline consist of mainbody detection, feature extraction and vector search. At last, we introduce classification base on inference model. 

- [CONVERT TRAINING MODEL TO INFERENCE MODEL](#CONVERT)
    - [Convert feature extraction model to inference model](#Convert_feature_extraction)
    - [Convert classification model to inference model](#Convert_class)

- [MAINBODY DETECTION MODEL INFERENCE](#DETECTION_MODEL_INFERENCE)

- [FEATURE EXTRACTION MODEL INFERENCE](#FEATURE_EXTRACTION_MODEL_INFERENCE)

- [CONCATENATION OF MAINBODY DETECTION, FEATURE EXTRACTION AND VECTOR SEARCH](#CONCATENATION)

- [CLASSIFICATION MODEL INFERENCE](#CLASSIFICATION)

<a name="CONVERT"></a>
## CONVERT TRAINING MODEL TO INFERENCE MODEL
<a name="Convert_feature_extraction"></a>
### Convert feature extraction model to inference model
First please enter the root folder of PaddleClas. Download the product feature extraction model:
```shell script
wget -P ./product_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained.pdparams
```

The above model is trained on AliProduct with ResNet50_vd as backbone. To convert the trained model into an inference model, just run the following command:
```
# -c Set the training algorithm yml configuration file
# -o Set optional parameters
# Global.pretrained_model parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
# Global.save_inference_dir Set the address where the converted model will be saved.

python3.7 tools/export_model.py -c ppcls/configs/Products/ResNet50_vd_Aliproduct.yaml -o Global.pretrained_model=./product_pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained -o Global.save_inference_dir=./deploy/models/product_ResNet50_vd_aliproduct_v1.0_infer
```

When converting to an inference model, the configuration file used is the same as the configuration file used during training. In addition, you also need to set the `Global.pretrained_model` parameter in the configuration file.
After the conversion is successful, there are three files in the model save directory:
``` 
├── product_ResNet50_vd_aliproduct_v1.0_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="Convert_class"></a>
### Convert classification model to inference model

Download the pretrained model:
``` shell script
wget -P ./cls_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams
```

the model is trained on ImageNet with ResNet50_vd as backbone, using config file `ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml`.
The model can be converted to the inference model in the same way as the feature extraction model, as follows:
```
# -c Set the training algorithm yml configuration file
# -o Set optional parameters
# Global.pretrained_model parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
# Global.save_inference_dir Set the address where the converted model will be saved.

python3.7 tools/export_model.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml -o Global.pretrained_model=./cls_pretrain/ResNet50_vd_pretrained -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer
```

After the conversion is successful, there are three files in the model save directory:
```
├── class_ResNet50_vd_ImageNet_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="DETECTION_MODEL_INFERENCE"></a>
## MAINBODY DETECTION MODEL INFERENCE

The following will introduce the mainbody detection model inference. All the command should be run under `deploy` folder of PaddleClas:
```shell script
cd deploy
```

For mainbody detection model inference, you can execute the following commands:

```shell script
mkdir -p models
cd models
# download mainbody detection inference model
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar && tar -xf ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
cd ..
# predict
python3.7 python/predict_det.py -c configs/inference_det.yaml
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
```shell script
python3.7 python/predict_det.py -c configs/inference_det.yaml -o Global.infer_imgs=images/anmuxi.jpg
```

If you want to use the CPU for prediction, you can switch value of `use_gpu` in config file to `False`. Or you can execute the command as follows
```
python3.7 python/predict_det.py -c configs/inference_det.yaml  -o Global.use_gpu=False
```

<a name="FEATURE_EXTRACTION_MODEL_INFERENCE"></a>
### FEATURE EXTRACTION MODEL INFERENCE

The following will introduce the feature extraction model inference. All the command should be run under `deploy` folder of PaddleClas:
```shell script
cd deploy
```
For feature extraction model inference, you can execute the following commands:
```shell script
mkdir -p models
cd models
# download feature extraction inference model
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/product_ResNet50_vd_aliproduct_v1.0_infer.tar && tar -xf product_ResNet50_vd_aliproduct_v1.0_infer.tar
cd ..
# predict
python3.7 python/predict_rec.py -c configs/inference_rec.yaml
```
You can get a 512-dim feature printed in the command line.

If you want to extract feature of another image, you can change the value of `infer_imgs` in `configs/inference_rec.yaml`, 
or you can use `-o Global.infer_imgs` argument. For example, if you want to try `images/anmuxi.jpg`:
```shell script
python3.7 python/predict_rec.py -c configs/inference_rec.yaml -o Global.infer_imgs=images/anmuxi.jpg
```

If you want to use the CPU for prediction, you can switch value of `use_gpu` in config file to `False`. Or you can execute the command as follows
```
python3.7 python/predict_rec.py -c configs/inference_rec.yaml  -o Global.use_gpu=False
```

<a name="CONCATENATION"></a>
## CONCATENATION OF MAINBODY DETECTION, FEATURE EXTRACTION AND VECTOR SEARCH
 Please refer to [Quick Start of Recognition](./tutorials/quick_start_recognition_en.md)

<a name="CLASSIFICATION"></a>
### CLASSIFICATION MODEL INFERENCE
The following will introduce the classification model inference. All the command should be run under `deploy` folder of PaddleClas:
```shell script
cd deploy
```

For classification model inference, you can execute the following commands:

```shell script
python3.7 python/predict_cls.py -c configs/inference_cls.yaml
```
If you want to try another image, you can change the value of `infer_imgs` in `configs/inference_cls.yaml`, 
or you can use `-o Global.infer_imgs` argument. For example, if you want to try `images/ILSVRC2012_val_00010010.jpeg`:
```shell script
python3.7 python/predict_cls.py -c configs/inference_cls.yaml -o Global.infer_imgs=images/ILSVRC2012_val_00010010.jpeg

```

If you want to use the CPU for prediction, you can switch value of `use_gpu` in config file to `False`. Or you can execute the command as follows
```
python3.7 python/predict_cls.py -c configs/inference_cls.yaml  -o Global.use_gpu=False
```