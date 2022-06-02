English|[Simplified Chinese](../../zh_CN/algorithm_introduction/reid.md)
# ReID pedestrian re-identification

### 1. Introduction to algorithms/application scenarios

Pedestrian re-identification (Person re-identification), also known as pedestrian re-identification, is the use of [computer vision](https://baike.baidu.com/item/computer vision/2803351) technology to judge [image](https://baike .baidu.com/item/image/773234) or whether there is a [technique] of a particular pedestrian in the video sequence (https://baike.baidu.com/item/technique/13014499). Widely regarded as a sub-problem of [Image Retrieval](https://baike.baidu.com/item/image_retrieval/1150910). Given a surveillance pedestrian image, retrieve the pedestrian image across devices. It aims to make up for the visual limitations of fixed cameras, and can be combined with [pedestrian detection](https://baike.baidu.com/item/pedestrian detection/20590256)/pedestrian tracking technology, which can be widely used in [intelligent video surveillance] ](https://baike.baidu.com/item/intelligent video surveillance/10717227), intelligent security and other fields.

### 2. Introduction to ReID strong-baseline algorithm

In the past person re-identification methods, the feature extraction module is used to extract the global or multi-granularity features of the image, and then a high-dimensional feature vector is obtained through the fusion module. Use the classification head to map the feature vector into the probability of each category during training to optimize the entire model in the way of classification tasks; directly use the high-dimensional feature vector as an image descriptor in the retrieval library during testing or inference. search to get the search results. The ReID strong-baseline algorithm proposes several methods to effectively optimize training and retrieval to improve the overall model performance.

#### 2.1 Principle of ReID strong-baseline algorithm

Paper source: [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)

<img src="../../images/reid/strong-baseline.jpg" width="50%">

Principle introduction: The author mainly uses the following optimization methods

1. Warmup, let the learning rate gradually increase at the beginning of training and then start to decrease, which is conducive to finding better parameters during gradient descent.
2. Random erasing augmentation, random area erasing, enhances the generalization of the model.
3. Label smoothing, label smoothing, enhance the generalization of the model.
4. Last stride=1, cancel the downsampling of the last stage of the feature extraction module, increase the resolution of the output feature map to retain more details and enhance the classification ability of the model.
5. BNNeck, before the feature vector is input into the classification head, it goes through BNNeck, so that the feature vector becomes a normal distribution, which reduces the difficulty of optimizing ID Loss and TripLetLoss at the same time.
6. Center loss, give each category a learnable cluster center feature, and make the intra-class features close to the cluster center during training to reduce intra-class differences and increase inter-class differences.
7. Reranking, consider whether the neighboring candidate objects of the retrieved image also contain retrieval targets during retrieval, so as to optimize the distance matrix and finally improve the retrieval accuracy.

#### 2.2a Quick start

The quick start chapter mainly takes the `softmax_triplet_with_center.yaml` configuration and trained model file as an example to test on the Market1501 dataset.

1. Download the [Market-1501-v15.09.15.zip](https://pan.baidu.com/s/1ntIi2Op?_at_=1654142245770) dataset, extract it to `PaddleClas/dataset/`, and organize it as follows File structure:

   ```shell
   PaddleClas/dataset/market1501
   └── Market-1501-v15.09.15/
   ├── bounding_box_test/
   ├── bounding_box_train/
   ├── gt_bbox/
   ├── gt_query/
   ├── query/
   ├── generate_anno.py
   ├── bounding_box_test.txt
   ├── bounding_box_train.txt
   ├── query.txt
   └── readme.txt
   ```

2. Download [reid_strong_baseline_softmax_with_center.epoch_120.pdparams](reid_strong_baseline_softmax_with_center.epoch_120.pdparams) to `PaddleClas/pretrained_models` folder

   ```shell
   cd PaddleClas
   mkdir pretrained_models
   cd pretrained_models
   wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/pretrain/reid_strong_baseline_softmax_with_center.epoch_120.pdparams
   cd..
   ```

3. Use the downloaded `softmax_triplet_with_center.pdparams` to test on the Market1501 dataset

   ```shell
   python3.7 tools/eval.py \
   -c ppcls/configs/reid/strong_baseline/baseline.yaml \
   -o Global.pretrained_model="pretrained_models/reid_strong_baseline_softmax_with_center.epoch_120"
   ```

4. View the output result

   ```log
   ...
   [2022/06/02 03:08:07] ppcls INFO: gallery feature calculation process: [0/125]
   [2022/06/02 03:08:11] ppcls INFO: gallery feature calculation process: [20/125]
   [2022/06/02 03:08:15] ppcls INFO: gallery feature calculation process: [40/125]
   [2022/06/02 03:08:19] ppcls INFO: gallery feature calculation process: [60/125]
   [2022/06/02 03:08:23] ppcls INFO: gallery feature calculation process: [80/125]
   [2022/06/02 03:08:27] ppcls INFO: gallery feature calculation process: [100/125]
   [2022/06/02 03:08:31] ppcls INFO: gallery feature calculation process: [120/125]
   [2022/06/02 03:08:32] ppcls INFO: Build gallery done, all feat shape: [15913, 2048], begin to eval..
   [2022/06/02 03:08:33] ppcls INFO: query feature calculation process: [0/27]
   [2022/06/02 03:08:36] ppcls INFO: query feature calculation process: [20/27]
   [2022/06/02 03:08:38] ppcls INFO: Build query done, all feat shape: [3368, 2048], begin to eval..
   [2022/06/02 03:08:38] ppcls INFO: re_ranking=False
   [2022/06/02 03:08:39] ppcls INFO: [Eval][Epoch 0][Avg]recall1: 0.94270, recall5: 0.98189, mAP: 0.85799
   ```

   It can be seen that the metrics of the `reid_strong_baseline_softmax_with_center.epoch_120.pdparams` model provided by us on the Market1501 dataset are recall@1=0.94270, recall@5=0.98189, mAP=0.85799

#### 2.2b Model training/testing/inference

- Model training

  1. Download the [Market-1501-v15.09.15.zip](https://pan.baidu.com/s/1ntIi2Op?_at_=1654142245770) dataset, extract it to `PaddleClas/dataset/`, and organize it as follows File structure:

     ```shell
     PaddleClas/dataset/market1501
     └── Market-1501-v15.09.15/
     ├── bounding_box_test/
     ├── bounding_box_train/
     ├── gt_bbox/
     ├── gt_query/
     ├── query/
     ├── generate_anno.py
     ├── bounding_box_test.txt
     ├── bounding_box_train.txt
     ├── query.txt
     └── readme.txt
     ```

  2. Execute the following command to start training

     ```shell
     python3.7 tools/train.py -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml
     ```

     Note: Single card training takes about 1 hour.

- Model testing

  Assuming that the path of the model file to be tested is `./output/RecModel/latest.pdparams` , execute the following command to test

  ```shell
  python3.7 tools/eval.py \
  -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
  -o Global.pretrained_model="./output/RecModel/latest"
  ```

  Note: The address filled after `pretrained_model` does not need to be suffixed with `.pdparams`, it will be added automatically when the program is running.

- Model inference

  1. DownloadInference model and extract: [reid_srong_baseline_softmax_with_center.tar](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/inference/reid_srong_baseline_softmax_with_center.tar)

     ```shell
     cd PaddleClas/deploy
     wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/inference/reid_srong_baseline_softmax_with_center.tar
     tar xf reid_srong_baseline_softmax_with_center.tar
     ```

  2. Modify `PaddleClas/deploy/configs/inference_rec.yaml`. Change the field after `infer_imgs:` to any image in the query folder in Market1501; change the field after `rec_inference_model_dir:` to the path of the extracted reid_srong_baseline_softmax_with_center folder; change the preprocessing configuration under the `transform_ops` field Changed to preprocessing configuration under `Eval.Query.dataset` in `softmax_triplet_with_center.yaml`. As follows

     ```yaml
     Global:
       infer_imgs: "../dataset/market1501/Market-1501-v15.09.15/query/0294_c1s1_066631_00.jpg"
       rec_inference_model_dir: "./reid_srong_baseline_softmax_with_center"
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
           interpolation: 'bilinear'
           backend: "pil"
       - ToTensor:
       - Normalize:
           mean: [0.485, 0.456, 0.406]
           std: [0.229, 0.224, 0.225]

     RecPostProcess: null
     ```

  3. Execute the inference command

     ```shell
     python3.7 python/predict_rec.py -c ./configs/inference_rec.yaml
     ```

  4. Check the output result. The actual result is a vector of length 2048, which represents the feature vector obtained after the input image is transformed by the model.

     ```shell
     0294_c1s1_066631_00.jpg: [ 0.01806974 0.00476423 -0.00508293 ... 0.03925538 0.00377574
      -0.00849029]
     ```

### 3. Summary

#### 3.1 Method summary, comparison, etc.

The following table summarizes the accuracy metrics of the 3 configurations of ReID strong-baseline we provide on the Market1501 dataset,

| Profile | recall@1 | mAP | Reference recall@1 | Reference mAP |
| ------------------------ | -------- | ----- | ------------ | ------- |
| baseline.yaml | 88.21 | 74.12 | 87.7 | 74.0 |
| softmax.yaml | 94.18 | 85.76 | 94.1 | 85.7 |
| softmax_with_center.yaml | 94.19 | 85.80 | 94.1 | 85.7 |

Note: The above reference indicators are obtained by using the author's open source code to train on our equipment for many times. Due to different system environments, torch versions, and CUDA versions, there may be slight differences with the indicators provided by the author.

#### 3.2 Usage advice/FAQ

The Market1501 dataset is relatively small, so you can try to train multiple times to get the highest accuracy.

#### 4 References

1. [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
2. [michuanhaohao/reid-strong-baseline: Bag of Tricks and A Strong Baseline for Deep Person Re-identification (github.com)](https://github.com/michuanhaohao/reid-strong-baseline)
3. [Pedestrian Re-ID dataset Market1501 dataset _star_function blog-CSDN blog _market1501 dataset](https://blog.csdn.net/qq_39220334/article/details/121470106)
