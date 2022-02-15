# Quick Start of Recognition

This tutorial contains 3 parts: Environment Preparation, Image Recognition Experience, and Unknown Category Image Recognition Experience.

If the image category already exists in the image index database, then you can take a reference to chapter [Image Recognition Experience](#2)，to complete the progress of image recognition；If you wish to recognize unknow category image, which is not included in the index database，you can take a reference to chapter [Unknown Category Image Recognition Experience](#3)，to complete the process of creating an index to recognize it。

## Catalogue

* [1. Enviroment Preparation](#1)
* [2. Image Recognition Experience](#2)
  * [2.1 Download and Unzip the Inference Model and Demo Data](#2.1)
  * [2.2 Product Recognition and Retrieval](#2.2)
    * [2.2.1 Single Image Recognition](#2.2.1)
    * [2.2.2 Folder-based Batch Recognition](#2.2.2)
* [3. Unknown Category Image Recognition Experience](#3)
  * [3.1 Prepare for the new images and labels](#3.1)
  * [3.2 Build a new Index Library](#3.2)
  * [3.3 Recognize the Unknown Category Images](#3.3)


<a name="1"></a>
## 1. Enviroment Preparation

* Installation：Please take a reference to [Quick Installation ](../installation/)to configure the PaddleClas environment.

* Using the following command to enter Folder `deploy`. All content and commands in this section need to be run in folder `deploy`.

  ```
  cd deploy
  ```

<a name="2"></a>
## 2. Image Recognition Experience

The detection model with the recognition inference model for the 4 directions (Logo, Cartoon Face, Vehicle, Product), the address for downloading the test data and the address of the corresponding configuration file are as follows.

| Models Introduction       | Recommended Scenarios  | inference Model  | Predict Config File  | Config File to Build Index Database |
| ------------  | ------------- | -------- | ------- | -------- |
| Generic mainbody detection model | General Scenarios |[Model Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar) | - | - |
| Logo Recognition Model | Logo Scenario  |  [Model Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/logo_rec_ResNet50_Logo3K_v1.0_infer.tar) | [inference_logo.yaml](../../../deploy/configs/inference_logo.yaml) | [build_logo.yaml](../../../deploy/configs/build_logo.yaml) |
| Cartoon Face Recognition Model| Cartoon Face Scenario  | [Model Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/cartoon_rec_ResNet50_iCartoon_v1.0_infer.tar) | [inference_cartoon.yaml](../../../deploy/configs/inference_cartoon.yaml) | [build_cartoon.yaml](../../../deploy/configs/build_cartoon.yaml) |
| Vehicle Fine-Grained Classfication Model | Vehicle Scenario  |   [Model Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/vehicle_cls_ResNet50_CompCars_v1.0_infer.tar) | [inference_vehicle.yaml](../../../deploy/configs/inference_vehicle.yaml) | [build_vehicle.yaml](../../../deploy/configs/build_vehicle.yaml) |
| Product Recignition Model | Product Scenario  |  [Model Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/product_ResNet50_vd_aliproduct_v1.0_infer.tar) | [inference_product.yaml](../../../deploy/configs/inference_product.yaml) | [build_product.yaml](../../../deploy/configs/build_product.yaml) |
| Vehicle ReID Model | Vehicle ReID Scenario | [Model Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/vehicle_reid_ResNet50_VERIWild_v1.0_infer.tar) | - | - |

| Models Introduction       | Recommended Scenarios   | inference Model  | Predict Config File  | Config File to Build Index Database |
| ------------  | ------------- | -------- | ------- | -------- |
| Lightweight generic mainbody detection model | General Scenarios  |[Model Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar) | - | - |
| Lightweight generic recognition model | General Scenarios  | [Model Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar) | [inference_product.yaml](../../../deploy/configs/inference_product.yaml) | [build_product.yaml](../../../deploy/configs/build_product.yaml) |


Demo data in this tutorial can be downloaded here: [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/recognition_demo_data_en_v1.1.tar).


**Attention**
1. If you do not have wget installed on Windows, you can download the model by copying the link into your browser and unzipping it in the appropriate folder; for Linux or macOS users, you can right-click and copy the download link to download it via the `wget` command.
2. If you want to install `wget` on macOS, you can run the following command.
3. The predict config file of the lightweight generic recognition model and the config file to build index database are used for the config of product recognition model of server-side. You can modify the path of the model to complete the index building and prediction.

```shell
# install homebrew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)";
# install wget
brew install wget
```

3. If you want to isntall `wget` on Windows, you can refer to [link](https://www.cnblogs.com/jeshy/p/10518062.html). If you want to install `tar` on Windows, you can refer to [link](https://www.cnblogs.com/chooperman/p/14190107.html).


* You can download and unzip the data and models by following the command below

```shell
mkdir models
cd models
# Download and unzip the inference model
wget {Models download link} && tar -xf {Name of the tar archive}
cd ..

# Download the demo data and unzip
wget {Data download link} && tar -xf {Name of the tar archive}
```


<a name="2.1"></a>
### 2.1 Download and Unzip the Inference Model and Demo Data

Take the product recognition as an example, download the detection model, recognition model and product recognition demo data with the following commands.

```shell
mkdir models
cd models
# Download the generic detection inference model and unzip it
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar && tar -xf ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
# Download and unpack the inference model
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/product_ResNet50_vd_aliproduct_v1.0_infer.tar && tar -xf product_ResNet50_vd_aliproduct_v1.0_infer.tar
cd ..

# Download the demo data and unzip it
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/recognition_demo_data_en_v1.1.tar && tar -xf recognition_demo_data_en_v1.1.tar
```

Once unpacked, the `recognition_demo_data_v1.1` folder should have the following file structure.

```
├── recognition_demo_data_v1.1
│   ├── gallery_cartoon
│   ├── gallery_logo
│   ├── gallery_product
│   ├── gallery_vehicle
│   ├── test_cartoon
│   ├── test_logo
│   ├── test_product
│   └── test_vehicle
├── ...
```

here, original images to build index are in folder `gallery_xxx`, test images are in folder `test_xxx`. You can also access specific folder for more details.

The `models` folder should have the following file structure.

```
├── product_ResNet50_vd_aliproduct_v1.0_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── ppyolov2_r50vd_dcn_mainbody_v1.0_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

**Attention**
If you want to use the lightweight generic recognition model, you need to re-extract the features of the demo data and re-build the index. The way is as follows:

```shell
python3.7 python/build_gallery.py -c configs/build_product.yaml -o Global.rec_inference_model_dir=./models/general_PPLCNet_x2_5_lite_v1.0_infer
```

<a name="2.2"></a>
### 2.2 Product Recognition and Retrieval

Take the product recognition demo as an example to show the recognition and retrieval process (if you wish to try other scenarios of recognition and retrieval, replace the corresponding configuration file after downloading and unzipping the corresponding demo data and model to complete the prediction).

**Note:**  `faiss` is used as search library. The installation method is as follows：

```
pip install faiss-cpu==1.7.1post2
```

If error happens when using `import faiss`, please uninstall `faiss` and reinstall it, especially on `Windows`.

<a name="2.2.1"></a>

#### 2.2.1 Single Image Recognition

Run the following command to identify and retrieve the image `./recognition_demo_data_v1.1/test_product/daoxiangcunjinzhubing_6.jpg` for recognition and retrieval

```shell
# use the following command to predict using GPU.
python3.7 python/predict_system.py -c configs/inference_product.yaml
# use the following command to predict using CPU
python3.7 python/predict_system.py -c configs/inference_product.yaml -o Global.use_gpu=False
```


The image to be retrieved is shown below.

![](../../images/recognition/product_demo/query/daoxiangcunjinzhubing_6.jpg)


The final output is shown below.

```
[{'bbox': [287, 129, 497, 326], 'rec_docs': 'Daoxaingcun Golden Piggie Cake', 'rec_scores': 0.8309420347213745}, {'bbox': [99, 242, 313, 426], 'rec_docs': 'Daoxaingcun Golden Piggie Cake', 'rec_scores': 0.7245651483535767}]
```


where bbox indicates the location of the detected object, rec_docs indicates the labels corresponding to the label in the index dabase that are most similar to the detected object, and rec_scores indicates the corresponding confidence.


The detection result is also saved in the folder `output`, for this image, the visualization result is as follows.

![](../../images/recognition/product_demo/result/daoxiangcunjinzhubing_6_en.jpg)


<a name="2.2.2"></a>
#### 2.2.2 Folder-based Batch Recognition

If you want to predict the images in the folder, you can directly modify the `Global.infer_imgs` field in the configuration file, or you can also modify the corresponding configuration through the following `-o` parameter.

```shell
# using the following command to predict using GPU, you can append `-o Global.use_gpu=False` to predict using CPU.
python3.7 python/predict_system.py -c configs/inference_product.yaml -o Global.infer_imgs="./recognition_demo_data_v1.1/test_product/"
```


The results on the screen are shown as following.

```
...
[{'bbox': [37, 29, 123, 89], 'rec_docs': 'Chanel Handbag', 'rec_scores': 0.6163763999938965}, {'bbox': [153, 96, 235, 175], 'rec_docs': 'Chanel Handbag', 'rec_scores': 0.5279821157455444}]
[{'bbox': [735, 562, 1133, 851], 'rec_docs': 'Chanel Handbag', 'rec_scores': 0.5588355660438538}]
[{'bbox': [124, 50, 230, 129], 'rec_docs': 'Chanel Handbag', 'rec_scores': 0.6980369687080383}]
[{'bbox': [0, 0, 275, 183], 'rec_docs': 'Chanel Handbag', 'rec_scores': 0.5818190574645996}]
[{'bbox': [400, 1179, 905, 1537], 'rec_docs': 'Chanel Handbag', 'rec_scores': 0.9814301133155823}, {'bbox': [295, 713, 820, 1046], 'rec_docs': 'Chanel Handbag', 'rec_scores': 0.9496176242828369}, {'bbox': [153, 236, 694, 614], 'rec_docs': 'Chanel Handbag', 'rec_scores': 0.8395382761955261}]
[{'bbox': [544, 4, 1482, 932], 'rec_docs': 'Chanel Handbag', 'rec_scores': 0.5143815279006958}]
...
```

All the visualization results are also saved in folder `output`.


Furthermore, the recognition inference model path can be changed by modifying the `Global.rec_inference_model_dir` field, and the path of the index to the index databass can be changed by modifying the `IndexProcess.index_dir` field.


<a name="3"></a>
## 3. Recognize Images of Unknown Category

To recognize the image `./recognition_demo_data_v1.1/test_product/anmuxi.jpg`, run the command as follows:

```shell
python3.7 python/predict_system.py -c configs/inference_product.yaml -o Global.infer_imgs="./recognition_demo_data_v1.1/test_product/anmuxi.jpg"
```

The image to be retrieved is shown below.

![](../../images/recognition/product_demo/query/anmuxi.jpg)

The output is empty.

Since the index infomation is not included in the corresponding index databse, the recognition result is empty or not proper. At this time, we can complete the image recognition of unknown categories by constructing a new index database.

When the index database cannot cover the scenes we actually recognise, i.e. when predicting images of unknown categories, we need to add similar images of the corresponding categories to the index databasey, thus completing the recognition of images of unknown categories ，which does not require retraining.

<a name="3.1"></a>
### 3.1 Prepare for the new images and labels

First, you need to copy the images which are similar with the image to retrieval to the original images for the index database. The command is as follows.

```shell
cp -r  ../docs/images/recognition/product_demo/gallery/anmuxi ./recognition_demo_data_/gallery_product/gallery/
```

Then you need to create a new label file which records the image path and label information. Use the following command to create a new file based on the original one.

```shell
# copy the file
cp recognition_demo_data_v1.1/gallery_product/data_file.txt recognition_demo_data_v1.1/gallery_product/data_file_update.txt
```

Then add some new lines into the new label file, which is shown as follows.

```
gallery/anmuxi/001.jpg	Anmuxi Ambrosial Yogurt
gallery/anmuxi/002.jpg	Anmuxi Ambrosial Yogurt
gallery/anmuxi/003.jpg	Anmuxi Ambrosial Yogurt
gallery/anmuxi/004.jpg	Anmuxi Ambrosial Yogurt
gallery/anmuxi/005.jpg	Anmuxi Ambrosial Yogurt
gallery/anmuxi/006.jpg	Anmuxi Ambrosial Yogurt
```

Each line can be splited into two fields. The first field denotes the relative image path, and the second field denotes its label. The `delimiter` is `tab` here.


<a name="3.2"></a>
### 3.2 Build a new Index Base Library

Use the following command to build the index to accelerate the retrieval process after recognition.

```shell
python3.7 python/build_gallery.py -c configs/build_product.yaml -o IndexProcess.data_file="./recognition_demo_data_v1.1/gallery_product/data_file_update.txt" -o IndexProcess.index_dir="./recognition_demo_data_v1.1/gallery_product/index_update"
```

Finally, the new index information is stored in the folder`./recognition_demo_data_v1.1/gallery_product/index_update`. Use the new index database for the above index.


<a name="3.3"></a>
### 3.3 Recognize the Unknown Category Images

To recognize the image `./recognition_demo_data_v1.1/test_product/anmuxi.jpg`, run the command as follows.

```shell
# using the following command to predict using GPU, you can append `-o Global.use_gpu=False` to predict using CPU.
python3.7 python/predict_system.py -c configs/inference_product.yaml -o Global.infer_imgs="./recognition_demo_data_v1.1/test_product/anmuxi.jpg" -o IndexProcess.index_dir="./recognition_demo_data_v1.1/gallery_product/index_update"
```

The output is as follows:

```
[{'bbox': [243, 80, 523, 522], 'rec_docs': 'Anmuxi Ambrosial Yogurt', 'rec_scores': 0.5570770502090454}]
```

The final recognition result is `Anmuxi Ambrosial Yogurt`, which is corrrect, the visualization result is as follows.

![](../../images/recognition/product_demo/result/anmuxi_en.jpg)
</div>
