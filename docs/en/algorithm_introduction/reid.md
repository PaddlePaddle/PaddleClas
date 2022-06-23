English | [简体中文](../../zh_CN/algorithm_introduction/reid.md)

# ReID pedestrian re-identification

## Table of contents

- [1. Introduction to algorithms/application scenarios](#1-introduction-to-algorithmsapplication-scenarios)
- [2. ReID algorithm](#2-reid-algorithm)
  - [2.1 ReID strong-baseline](#21-reid-strong-baseline)
    - [2.1.1 Principle introduction](#211-principle-introduction)
    - [2.1.2 Accuracy Index](#212-accuracy-index)
    - [2.1.3 Data Preparation](#213-data-preparation)
    - [2.1.4 Model training](#214-model-training)
- [3. Model evaluation and inference deployment](#3-model-evaluation-and-inference-deployment)
  - [3.1 Model Evaluation](#31-model-evaluation)
  - [3.2 Model Inference and Deployment](#32-model-inference-and-deployment)
    - [3.2.1 Inference model preparation](#321-inference-model-preparation)
    - [3.2.2 Inference based on Python prediction engine](#322-inference-based-on-python-prediction-engine)
    - [3.2.3 Inference based on C++ prediction engine](#323-inference-based-on-c-prediction-engine)
  - [3.3 Service Deployment](#33-service-deployment)
  - [3.4 Lite deployment](#34-lite-deployment)
  - [3.5 Paddle2ONNX model conversion and prediction](#35-paddle2onnx-model-conversion-and-prediction)
- [4. Summary](#4-summary)
  - [4.1 Method summary and comparison](#41-method-summary-and-comparison)
  - [4.2 Usage advice/FAQ](#42-usage-advicefaq)
- [4. References](#4-references)

### 1. Introduction to algorithms/application scenarios

Pedestrian re-identification, also known as pedestrian re-identification, is a technology that uses computer vision technology to determine whether there is a specific pedestrian in an image or video sequence. Widely regarded as a sub-problem of image retrieval. Given a surveillance pedestrian image, retrieve the pedestrian image across devices. It is designed to make up for the visual limitations of fixed cameras, and can be combined with pedestrian detection/pedestrian tracking technology, which can be widely used in intelligent video surveillance, intelligent security and other fields.

The common person re-identification method extracts the local/global, single-granularity/multi-granularity features of the input image through the feature extraction module, and then obtains a high-dimensional feature vector through the fusion module. Use the classification head to convert the feature vector into the probability of each category during training to optimize the feature extraction model in the way of classification tasks; directly use the high-dimensional feature vector as the image description vector in the retrieval vector library during testing or inference search to get the search results. The ReID strong-baseline algorithm proposes several methods to effectively optimize training and retrieval to improve the overall model performance.
<img src="../../images/reid/reid_overview.jpg" align="middle">

### 2. ReID algorithm

#### 2.1 ReID strong-baseline

Paper source: [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)

<img src="../../images/reid/strong-baseline.jpg" width="80%">

##### 2.1.1 Principle introduction

Based on the commonly used person re-identification model based on ResNet50, the author explores and summarizes the following effective and applicable optimization methods, which greatly improves the indicators on multiple person re-identification datasets.

1. Warmup: At the beginning of training, let the learning rate gradually increase from a small value and then start to decrease, which is conducive to the stability of gradient descent optimization, so as to find a better parameter model.
2. Random erasing augmentation: Random area erasing, which improves the generalization ability of the model through data augmentation.
3. Label smoothing: Label smoothing to improve the generalization ability of the model.
4. Last stride=1: Set the downsampling of the last stage of the feature extraction module to 1, increase the resolution of the output feature map to retain more details and improve the classification ability of the model.
5. BNNeck: Before the feature vector is input to the classification head, it goes through BNNeck, so that the feature obeys the normal distribution on the surface of the hypersphere, which reduces the difficulty of optimizing IDLoss and TripLetLoss at the same time.
6. Center loss: Give each category a learnable cluster center, and make the intra-class features close to the cluster center during training to reduce intra-class differences and increase inter-class differences.
7. Reranking: Consider the neighbor candidates of the query image during retrieval, optimize the distance matrix according to whether the neighbor images of the candidate object also contain the query image, and finally improve the retrieval accuracy.

##### 2.1.2 Accuracy Index

The following table summarizes the accuracy metrics of the 3 configurations of the recurring ReID strong-baseline on the Market1501 dataset,

| Configuration file               | recall@1(\%) | mAP(\%) | Refer to recall@1(\%) | Refer to mAP(\%) | Pre-trained model download                                                                                                                   | Inference Model Download Address                                                                                                    |
| -------------------------------- | ------------ | ------- | --------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| baseline.yaml                    | 88.45        | 74.37   | 87.7                  | 74.0             | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/pretrain/baseline_pretrained.pdparams)                    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/inference/baseline_infer.tar)                    |
| softmax_triplet.yaml             | 94.29        | 85.57   | 94.1                  | 85.7             | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/pretrain/softmax_triplet_pretrained.pdparams)             | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/inference/softmax_triplet_infer.tar)             |
| softmax_triplet_with_center.yaml | 94.50        | 85.82   | 94.5                  | 85.9             | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/pretrain/softmax_triplet_with_center_pretrained.pdparams) | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/inference/softmax_triplet_with_center_infer.tar) |

Note: The above reference indicators are obtained by using the author's open source code to train on our equipment for many times. Due to different system environments, torch versions, and CUDA versions, there may be slight differences with the indicators provided by the author.

Next, we mainly take the `softmax_triplet_with_center.yaml` configuration and trained model file as an example to show the process of training, testing, and inference on the Market1501 dataset.

##### 2.1.3 Data Preparation

Download the [Market-1501-v15.09.15.zip](https://pan.baidu.com/s/1ntIi2Op?_at_=1654142245770) dataset, extract it to `PaddleClas/dataset/`, and organize it into the following file structure :

  ```shell
  PaddleClas/dataset/market1501
  └── Market-1501-v15.09.15/
      ├── bounding_box_test/     # gallery set pictures
      ├── bounding_box_train/    # training set image
      ├── gt_bbox/
      ├── gt_query/
      ├── query/                 # query set image
      ├── generate_anno.py
      ├── bounding_box_test.txt  # gallery set path
      ├── bounding_box_train.txt # training set path
      ├── query.txt              # query set path
      └── readme.txt
  ```

##### 2.1.4 Model training

1. Execute the following command to start training

    Single card training:
    ```shell
    python3.7 tools/train.py -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml
    ```

    Doka training:

    For Doka training, you need to modify the sampler field of the training configuration as follows:
    ```yaml
    sampler:
      name: PKSampler
      batch_size: 64
      sample_per_id: 4
      drop_last: False
      sample_method: id_avg_prob
      shuffle: True
    ```
    Then execute the following command(example of 4 gpus below):
    ```shell
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py \
    -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml
    ```
    Note: Single card training takes about 1 hour.

2. View training logs and saved model parameter files

    During the training process, indicator information such as loss will be printed on the screen in real time, and the log file `train.log`, model parameter file `*.pdparams`, optimizer parameter file `*.pdopt` and other contents will be saved to `Global.output_dir` `Under the specified folder, the default is under the `PaddleClas/output/RecModel/` folder.

### 3. Model evaluation and inference deployment

#### 3.1 Model Evaluation

Prepare the `*.pdparams` model parameter file for evaluation. You can use the trained model or the model saved in [2.1.4 Model training] (#214-model training).

- Take the `latest.pdparams` saved during training as an example, execute the following command to evaluate.

  ```shell
  python3.7 tools/eval.py \
  -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
  -o Global.pretrained_model="./output/RecModel/latest"
  ```

- to train wellTake the model as an example, download [softmax_triplet_with_center_pretrained.pdparams](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/pretrain/softmax_triplet_with_center_pretrained.pdparams) to `PaddleClas/ In the pretrained_models` folder, execute the following command to evaluate.

  ```shell
  # download model
  cd PaddleClas
  mkdir pretrained_models
  cd pretrained_models
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/pretrain/softmax_triplet_with_center_pretrained.pdparams
  cd..
  # Evaluate
  python3.7 tools/eval.py \
  -c ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
  -o Global.pretrained_model="pretrained_models/softmax_triplet_with_center_pretrained"
  ```
  Note: The address filled after `pretrained_model` does not need to be suffixed with `.pdparams`, it will be added automatically when the program is running.

- View output results
  ```log
  ...
  ...
  ppcls INFO: unique_endpoints {''}
  ppcls INFO: Found /root/.paddleclas/weights/resnet50-19c8e357_torch2paddle.pdparams
  ppcls INFO: gallery feature calculation process: [0/125]
  ppcls INFO: gallery feature calculation process: [20/125]
  ppcls INFO: gallery feature calculation process: [40/125]
  ppcls INFO: gallery feature calculation process: [60/125]
  ppcls INFO: gallery feature calculation process: [80/125]
  ppcls INFO: gallery feature calculation process: [100/125]
  ppcls INFO: gallery feature calculation process: [120/125]
  ppcls INFO: Build gallery done, all feat shape: [15913, 2048], begin to eval..
  ppcls INFO: query feature calculation process: [0/27]
  ppcls INFO: query feature calculation process: [20/27]
  ppcls INFO: Build query done, all feat shape: [3368, 2048], begin to eval..
  ppcls INFO: re_ranking=False
  ppcls INFO: [Eval][Epoch 0][Avg]recall1: 0.94507, recall5: 0.98248, mAP: 0.85827
  ```
  The default evaluation log is saved in `PaddleClas/output/RecModel/eval.log`. You can see that the evaluation metrics of the `softmax_triplet_with_center_pretrained.pdparams` model we provided on the Market1501 dataset are recall@1=0.94507, recall@5 =0.98248, mAP=0.85827

#### 3.2 Model Inference and Deployment

##### 3.2.1 Inference model preparation

You can convert the model file saved during training into an inference model and inference, or use the converted inference model we provide for direct inference
  - Convert the model file saved during the training process to an inference model, also take `latest.pdparams` as an example, execute the following command to convert
    ```shell
    python3.7 tools/export_model.py \
    -c ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
    -o Global.pretrained_model="output/RecModel/latest" \
    -o Global.save_inference_dir="./deploy/softmax_triplet_with_center_infer"
    ```

  - Or download and unzip the inference model we provide
    ```shell
    cd PaddleClas/deploy
    wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/inference/softmax_triplet_with_center_infer.tar
    tar xf softmax_triplet_with_center_infer.tar
    cd ../
    ```

##### 3.2.2 Inference based on Python prediction engine

  1. Modify `PaddleClas/deploy/configs/inference_rec.yaml`. Change the field after `infer_imgs:` to any image path under the query folder in Market1501 (the configuration below uses the path of the `0294_c1s1_066631_00.jpg` image); change the field after `rec_inference_model_dir:` to extract it softmax_triplet_with_center_infer folder path; change the preprocessing configuration under the `transform_ops` field to the preprocessing configuration under `Eval.Query.dataset` in `softmax_triplet_with_center.yaml`. As follows

      ```yaml
      Global:
        infer_imgs: "../dataset/market1501/Market-1501-v15.09.15/query/0294_c1s1_066631_00.jpg"
        rec_inference_model_dir: "./softmax_triplet_with_center_infer"
        batch_size: 1
        use_gpu: False
        enable_mkldnn: True
        cpu_num_threads: 10
        enable_benchmark: True
        use_fp16: False
        ir_optim: True
        use_tensorrt: False
        gpu_mem: 8000
        enable_profile: False

      RecPreProcess:
        transform_ops:
          -ResizeImage:
              size: [128, 256]
              return_numpy: False
              interpolation: "bilinear"
              backend: "pil"
          - ToTensor:
          - Normalize:
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]

      RecPostProcess: null
      ```

  2. Execute the inference command

       ```shell
       cd PaddleClas/deploy/
       python3.7 python/predict_rec.py -c ./configs/inference_rec.yaml
       ```

  3. Check the output result, the actual result is a vector of length 2048, which represents the feature vector obtained after the input image is transformed by the model

       ```log
       0294_c1s1_066631_00.jpg: [ 0.01806974 0.00476423 -0.00508293 ... 0.03925538 0.00377574
        -0.00849029]
       ```
        The output vector for inference is stored in the `result_dict` variable in [predict_rec.py](../../../deploy/python/predict_rec.py#L134-L135).

  4. For batch prediction, change the path after `infer_imgs:` in the configuration file to a folder, such as `../dataset/market1501/Market-1501-v15.09.15/query`, it will predict and output The feature vector of all images under query.

##### 3.2.3 Inference based on C++ prediction engine

PaddleClas provides an example of inference based on C++ prediction engine, you can refer to [C++ prediction](../inference_deployment/cpp_deploy_en.md) to complete the corresponding inference deployment. If you are using the Windows platform, you can refer to the Visual Studio 2019 Community CMake Compilation Guide to complete the corresponding prediction library compilation and model prediction work.

#### 3.3 Service Deployment

Paddle Serving provides high-performance, flexible and easy-to-use industrial-grade online inference services. Paddle Serving supports RESTful, gRPC, bRPC and other protocols, and provides inference solutions in a variety of heterogeneous hardware and operating system environments. For more introduction to Paddle Serving, please refer to the Paddle Serving code repository.

PaddleClas provides an example of model serving deployment based on Paddle Serving. You can refer to [Model serving deployment](../inference_deployment/paddle_serving_deploy_en.md) to complete the corresponding deployment work.

#### 3.4 Lite deployment

Paddle Lite is a high-performance, lightweight, flexible and easily extensible deep learning inference framework, positioned to support mobileMultiple hardware platforms including client, embedded and server. For more introduction to Paddle Lite, please refer to the Paddle Lite code repository.

PaddleClas provides an example of deploying models based on Paddle Lite. You can refer to [Deployment](../inference_deployment/paddle_lite_deploy_en.md) to complete the corresponding deployment.

#### 3.5 Paddle2ONNX model conversion and prediction

Paddle2ONNX supports converting PaddlePaddle model format to ONNX model format. The deployment of Paddle models to various inference engines can be completed through ONNX, including TensorRT/OpenVINO/MNN/TNN/NCNN, as well as other inference engines or hardware that support the ONNX open source format. For more information about Paddle2ONNX, please refer to the Paddle2ONNX code repository.

PaddleClas provides an example of converting an inference model to an ONNX model and making inference prediction based on Paddle2ONNX. You can refer to [Paddle2ONNX model conversion and prediction](../../../deploy/paddle2onnx/readme.md) to complete the corresponding deployment work.

### 4. Summary

#### 4.1 Method summary and comparison

The above algorithm can be quickly migrated to most ReID models, which can further improve the performance of ReID models.

#### 4.2 Usage advice/FAQ

The Market1501 dataset is relatively small, so you can try to train multiple times to get the highest accuracy.

### 4. References

1. [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
2. [michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)
3. [Pedestrian Re-ID dataset Market1501 dataset _star_function blog-CSDN blog _market1501 dataset](https://blog.csdn.net/qq_39220334/article/details/121470106)
4. [Deep Learning for Person Re-identification: A Survey and Outlook](https://arxiv.org/abs/2001.04193)
