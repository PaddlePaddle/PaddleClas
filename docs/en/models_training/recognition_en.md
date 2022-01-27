# Image Recognition



Image recognition, in PaddleClas, means that the system is able to recognize the label of a given query image. Broadly speaking, image classification falls under image recognition. But unlike ordinary image recognition, it can only discriminate the learned categories and require retraining to add new ones. The image recognition in PaddleClas, however, only need to update the corresponding search library to identify the category of the unfamiliar images without retraining the model, which not only significantly promotes the usability of the recognition system but also reduces the demand for model updates, facilitating users' deployment of the application.

For an image to be queried, the image recognition process in PaddleClas is divided into three main parts:

1. Mainbody Detection: for a given query image, the mainbody detector first identifies the object, thus removing useless background information to improve the recognition accuracy.
2. Feature Extraction: for each candidate region of mainbody detection, feature extraction is performed by the feature model
3. Vector Search: the extracted features are compared with the vectors in the feature gallery for similarity to obtain their label information

The feature gallery is built in advance using the labeled image datasets. The complete image recognition system is shown in the figure below.

![img](../../images/structure.jpg)

To experience the whole image recognition system, or learn  how to build a feature gallery, please refer to [Quick Start of Image Recognition](../quick_start/quick_start_recognition_en.md), which explains the overall application process. The following parts expound on the training part of the above three steps.

Please first refer to the [Installation Guide](../installation/install_paddleclas_en.md) to configure the runtime environment.

## Catalogue

- [1. Mainbody Detection](#1)
- [2. Feature Model Training](#2)
  - [2.1. Data Preparation](#2.1)
  - [2. 2 Single GPU-based Training and Evaluation](#2.2)
    - [2.2.1 Model Training](#2.2.1)
    - [2.2.2 Resume Training](#2.2.2)
    - [2.2.3 Model Evaluation](#2.2.3)
  - [2.3 Export Inference Model](#2.3)
- [3.  Vector Search](#3)
- [4. Basic Knowledge](#4)

<a name="1"></a>  

## 1. Mainbody Detection

The mainbody detection training is based on [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop), the only difference is that all the detection boxes in the mainbody detection task belong to the foreground, but it is necessary to modify  `category_id` of the detection box in the annotation file to 1, while changing the `categories` mapping table in the whole annotation file to the following format, i.e., the whole category mapping table contains only `foreground`.

```
[{u'id': 1, u'name': u'foreground', u'supercategory': u'foreground'}]
```

For more information about the training method of mainbody detection, please refer to: [PaddleDetection Training Tutorial](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md#4-训练).

For more information on the introduction and download of the model provided in PaddleClas for body detection, please refer to: [PaddleDetection Tutorial](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/image_recognition_pipeline/mainbody_detection.md).

<a name="2"></a>  

## 2. Feature Model Training

<a name="2.1"></a>  

### 2.1 Data Preparation

- Go to PaddleClas directory.

```
## linux or mac， $path_to_PaddleClas indicates the root directory of PaddleClas，which the user needs to modify according to their real directory
cd $path_to_PaddleClas
```

- Go to the  `dataset`. which the user needs to modify according to their real directory [CUB_200_2011](http://vision.ucsd.edu/sites/default/files/WelinderEtal10_CUB-200.pdf), which is a fine grid dataset with 200 different types of birds. Firstly, we need to download the dataset. For download, please refer to [Official Website](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

```shell
# linux or mac
cd dataset

# Copy the downloaded data into a directory.
cp {Data storage path}/CUB_200_2011.tgz .

# Unzip
tar -xzvf CUB_200_2011.tgz

# Go to CUB_200_2011
cd CUB_200_2011
```

When using the dataset for image retrieval, we usually use the first 100 classes as the training set, and the last 100 classes as the testing set, so we need to process those data so as to adapt the model training of image retrieval.

```shell
# Create train and test directories
mkdir train && mkdir test

# Divide data into training set with the first 100 classes and testing set with the last 100 classes.
ls images | awk -F "." '{if(int($1)<101)print "mv images/"$0" train/"int($1)}' | sh
ls images | awk -F "." '{if(int($1)>100)print "mv images/"$0" test/"int($1)}' | sh

# Generate train_list and test_list
tree -r -i -f train | grep jpg | awk -F "/" '{print $0" "int($2) " "NR}' > train_list.txt
tree -r -i -f test | grep jpg | awk -F "/" '{print $0" "int($2) " "NR}' > test_list.txt
```

So far, we have the training set (in the `train` catalog), testing set (in the `test` catalog), `train_list.txt` and `test_list.txt` of `CUB_200_2011`.

After data preparation, the `train` directory of `CUB_200_2011` should be：

```
├── 1
│   ├── Black_Footed_Albatross_0001_796111.jpg
│   ├── Black_Footed_Albatross_0002_55.jpg
 ...
├── 10
│   ├── Red_Winged_Blackbird_0001_3695.jpg
│   ├── Red_Winged_Blackbird_0005_5636.jpg
...
```

`train_list.txt` should be：

```
train/99/Ovenbird_0137_92639.jpg 99 1
train/99/Ovenbird_0136_92859.jpg 99 2
train/99/Ovenbird_0135_93168.jpg 99 3
train/99/Ovenbird_0131_92559.jpg 99 4
train/99/Ovenbird_0130_92452.jpg 99 5
...
```

The separators are shown as spaces, and the meaning of those three columns of data are the directory, label and unique id of training sets.

The format of testing set is the same as the one of training set.

**Note**：

- When the gallery dataset and query dataset are the same, in order to remove the first data retrieved (the retrieved images themselves do not need to be evaluated), each data needs to correspond to a unique id for subsequent evaluation of metrics such as mAP, recall@1, etc. Please refer to [Introduction to image retrieval datasets](#Introduction to Image Retrieval Datasets) for the analysis of gallery datasets and query datasets, and [Image retrieval evaluation metrics](#Image Retrieval Evaluation Metrics) for the evaluation of mAP, recall@1, etc.

Back to `PaddleClas` root directory.

```shell
# linux or mac
cd ../../
```

<a name="2.2"></a>  

### 2.2 Single GPU-based Training and Evaluation

For training and evaluation on a single GPU, the `tools/train.py` and `tools/eval.py` scripts are recommended.

PaddleClas support training with VisualDL to visualize the metric. VisualDL is a visualization analysis tool of PaddlePaddle, provides a variety of charts to show the trends of parameters, and visualizes model structures, data samples, histograms of tensors, PR curves , ROC curves and high-dimensional data distributions. It enables users to understand the training process and the model structure more clearly and intuitively so as to optimize models efficiently. For more information, please refer to [VisualDL](../others/VisualDL_en.md).

<a name="2.2.1"></a>  

#### 2.2.1 Model Training

Once you have prepared the configuration file, you can start training the image retrieval task in the following way. the method used by PaddleClas to train the image retrieval is metric learning, referring to [metric learning](#metric learning) for more explanations.

```shell
# Single GPU
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Arch.Backbone.pretrained=True \
    -o Global.device=gpu
# Multi GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Arch.Backbone.pretrained=True \
    -o Global.device=gpu
```

`-c` is used to specify the path to the configuration file, and `-o` is used to specify the parameters that need to be modified or added, where `-o Arch.Backbone.pretrained=True` indicates that the Backbone part uses the pre-trained model. In addtion,`Arch.Backbone.pretrained`  can also specify the address of a specific model weight file, which needs to be replaced with the path to your own pre-trained model weight file when using it. `-o Global.device=gpu` indicates that the GPU is used for training. If you want to use a CPU for training, you need to set `Global.device` to `cpu`.

For more detailed training configuration, you can also modify the corresponding configuration file of the model directly. Refer to the [configuration document](config_description_en.md) for specific configuration parameters.

Run the above commands to check the output log, an example is as follows:

````
```
...
[Train][Epoch 1/50][Avg]CELoss: 6.59110, TripletLossV2: 0.54044, loss: 7.13154
...
[Eval][Epoch 1][Avg]recall1: 0.46962, recall5: 0.75608, mAP: 0.21238
...
```
````

The Backbone here is MobileNetV1, if you want to use other backbone, you can rewrite the parameter `Arch.Backbone.name`, for example by adding `-o Arch.Backbone.name={other Backbone}` to the command. In addition, as the input dimension of the `Neck` section differs between models, replacing a Backbone may require rewriting the input size here in a similar way to replacing the Backbone's name.

In the Training Loss section, [CELoss](../../../ppcls/loss/celoss.py) and [TripletLossV2](../../../ppcls/loss/triplet.py) are used here with the following configuration files:

```
Loss:
  Train:
    - CELoss:
        weight: 1.0
    - TripletLossV2:
        weight: 1.0
        margin: 0.5
```

The final total Loss is a weighted sum of all Losses, where weight defines the weight of a particular Loss in the final total. If you want to replace other Losses, you can also change the Loss field in the configuration file, for the currently supported Losses please refer to [Loss](../../../ppcls/loss).

<a name="2.2.2"></a>  

#### 2.2.2 Resume Training

If the training task is terminated for some reasons, it can be recovered by loading the checkpoints weights file and continue training:

```shell
# Single card
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.checkpoints="./output/RecModel/epoch_5" \
    -o Global.device=gpu
# Multi card
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.checkpoints="./output/RecModel/epoch_5" \
    -o Global.device=gpu
```

There is no need to modify the configuration file, just set the `Global.checkpoints` parameter when continuing training, indicating the path to the loaded breakpoint weights file, using this parameter will load both the saved checkpoints weights and information about the learning rate, optimizer, etc.

**Note**：

- The `-o Global.checkpoints` parameter need not contain the suffix name of the checkpoint weights file, the above training command will generate the breakpoint weights file as shown below during training, if you want to continue training from breakpoint `5` then the `Global.checkpoints` parameter just needs to be set to `". /output/RecModel/epoch_5"` and PaddleClas will automatically supplement the suffix name.

  ```
  output/
  └── RecModel
      ├── best_model.pdopt
      ├── best_model.pdparams
      ├── best_model.pdstates
      ├── epoch_1.pdopt
      ├── epoch_1.pdparams
      ├── epoch_1.pdstates
      .
      .
      .
  ```

<a name="2.2.3"></a>  

#### 2.2.3 Model Evaluation

Model evaluation can be carried out with the following commands.

```shell
# Single card
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.pretrained_model=./output/RecModel/best_model
# Multi card
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch tools/eval.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.pretrained_model=./output/RecModel/best_model
```

The above command will use `./configs/quick_start/MobileNetV1_retrieval.yaml` as a configuration file to evaluate the model obtained from the above training `./output/RecModel/best_model` for evaluation. You can also set up the evaluation by changing the parameters in the configuration file, or you can update the configuration with the `-o` parameter, as shown above.

Some of the configurable evaluation parameters are introduced as follows.

- `Arch.name`：the name of the model
- `Global.pretrained_model`：path to the pre-trained model file of the model to be evaluated, unlike `Global.Backbone.pretrained`,  the pre-trained model is the weight of the whole model instead of the Backbone only. When it is time to do model evaluation, the weights of the whole model need to be loaded.
- `Metric.Eval`：the metric to be evaluated, by default evaluates recall@1, recall@5, mAP. when you are not going to evaluate a metric, you can remove the corresponding trial marker from the configuration file; when you want to add a certain evaluation metric, you can also refer to [Metric](../../../ppcls/metric/metrics.py) section to add the relevant metric to the configuration file `Metric.Eval`.

**Note：**

- When loading the model to be evaluated, the path to the model file needs to be specified, but it is not necessary to include the file suffix, PaddleClas will automatically complete the `.pdparams` suffix, e.g. [2.2.2 Resume Training](#2.2.2).
- Metric learning are generally not evaluated for TopkAcc.

<a name="2.3"></a>

### 2.3 Export Inference Model

By exporting the inference model, PaddlePaddle supports the transformation of the trained model using prediction with inference engine.

```shell
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/MobileNetV1_retrieval.yaml \
    -o Global.pretrained_model=output/RecModel/best_model \
    -o Global.save_inference_dir=./inference
```

`Global.pretrained_model` is used to specify the model file path, which still does not need to contain the model file suffix (e.g.[2.2.2 Model Recovery Training](#2.2.2)). When executed, it will generate the `./inference` directory, which contains the `inference.pdiparams`,`inference.pdiparams.info`, and`inference.pdmodel` files.`Global.save_inference_dir` allows you to specify the path to export the inference model. The inference model saved here is truncated at the embedding feature level, i.e. the final output of the model is n-dimensional embedding features.

The above command will generate the model structure file (`inference.pdmodel`) and the model weights file (`inference.pdiparams`), which can then be used for inference using the inference engine. The process of inference using the inference model can be found in [Predictive inference based on the Python prediction engine](../inference_deployment/python_deploy_en.md).

<a name="3"></a>

## 3. Vector Search

Vector search in PaddleClas currently supports the following environments:

```
└── CPU
    ├── Linux
    ├── MacOS
    └── Windows
```

[Faiss](https://github.com/facebookresearch/faiss) is adopted as a search library, which is an efficient one for feature search and clustering. A variety of similarity search algorithms are integrated in this library to meet different scenarios. In PaddleClas, three search algorithms are supported.

- **HNSW32**: A graph indexing method boasts high retrieval accuracy and fast speed. However, the feature library only supports the function of adding images, not deleting image features. (Default method)
- **IVF**: An inverted index search method with fast speed but slightly lower precision. The feature library supports functions of adding and deleting image features.
- **FLAT**:  A violent search algorithm presenting the highest precision, but slower retrieval speed in face of large data volume. The feature library supports functions of adding and deleting image features.

See its detailed introduction in the [official document](https://github.com/facebookresearch/faiss/wiki).

`Faiss` can be installed as follows:

```
pip install faiss-cpu==1.7.1post2
```

If the above cannot be properly referenced, please `uninstall` and then  `install` again, especially when you are using`windows`.

<a name="4"></a>

## 4. Basic Knowledge

Image retrieval refers to a query image given a specific instance (e.g. a specific target, scene, item, etc.) that contains the same instance from a database image. Unlike image classification, image retrieval solves an open set problem where the training set may not contain the class of the image being recognised. The overall process of image retrieval is: firstly, the images are represented in a suitable feature vector, secondly, a nearest neighbour search is performed on these image feature vectors using Euclidean or Cosine distances to find similar images in the base, and finally, some post-processing techniques can be used to fine-tune the retrieval results and determine information such as the category of the image being recognised. Therefore, the key to determining the performance of an image retrieval algorithm lies in the goodness of the feature vectors corresponding to the images.

<a name="metric learning"></a>

- Metric Learning

Metric learning studies how to learn a distance function on a particular task so that the distance function can help nearest-neighbour based algorithms (kNN, k-means, etc.) to achieve better performance. Deep Metric Learning is a method of metric learning that aims to learn a mapping from the original features to a low-dimensional dense vector space (embedding space) such that similar objects on the embedding space are closer together using commonly used distance functions (Euclidean distance, cosine distance, etc.) ) on the embedding space, while the distances between objects of different classes are not close to each other. Deep metric learning has achieved very successful applications in the field of computer vision, such as face recognition, commodity recognition, image retrieval, pedestrian re-identification, etc. See [HERE](../algorithm_introduction/metric_learning_en.md) for detailed information.

<a name="Introduction to Image Retrieval Datasets"></a>

- Introduction to Image Retrieval Datasets
  - Training Dataset: used to train the model so that it can learn the image features of the collection.
  - Gallery Dataset: used to provide the gallery data for the image retrieval task. The gallery dataset can be the same as the training set or the test set, or different.
  - Test Set (Query Dataset): used to test the goodness of the model, usually each test image in the test set is extracted with features, and then matched with the features of the underlying data to obtain recognition results, and then the metrics of the whole test set are calculated based on the recognition results.

<a name="Image Retrieval Evaluation Metrics"></a>

- Image Retrieval Evaluation Metrics

  - recall: indicates the number of predicted positive cases with positive labels / the number of cases with positive labels
    - recall@1: Number of predicted positive cases in top-1 with positive label / Number of cases with positive label
    - recall@5: Number of all predicted positive cases in top-5 retrieved with positive label / Number of cases with positive label

  - mean Average Precision(mAP)
    - AP: AP refers to the average precision on different recall rates
    - mAP: Average of the APs for all images in the test set
