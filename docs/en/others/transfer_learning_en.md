# Transfer learning in image classification

Transfer learning is an important part of machine learning, which is widely used in various fields such as text and images. Here we mainly introduce transfer learning in the field of image classification, which is often called domain transfer, such as migration of the ImageNet classification model to the specified image classification task, such as flower classification.

---

## Catalogue

* [1. Hyperparameter search](#1)
  * [1.1 Grid search](#1.1)
  * [1.2 Bayesian search](#1.2)
* [2. Large-scale image classification](#2)
* [3. Reference](#3)

<a name='1'></a>
## 1. Hyperparameter search

ImageNet is the widely used dataset for image classification. A series of empirical hyperparameters have been summarized. High accuracy can be got using the hyperparameters. However, when applied in the specified dataset, the hyperparameters may not be optimal. There are two commonly used hyperparameter search methods that can be used to help us obtain better model hyperparameters.

<a name='1.1'></a>
### 1.1 Grid search

For grid search, which is also called exhaustive search, the optimal value is determined by finding the best solution from all solutions in the search space. The method is simple and effective, but when the search space is large, it takes huge computing resource.

<a name='1.2'></a>
### 1.2 Bayesian search

Bayesian search, which is also called Bayesian optimization, is realized by randomly selecting a group of hyperparameters in the search space. Gaussian process is used to update the hyperparameters, compute their expected mean and variance according to the performance of the previous hyperparameters. The larger the expected mean, the greater the probability of being close to the optimal solution. The larger the expected variance, the greater the uncertainty. Usually, the hyperparameter point with large expected mean is called `exporitation`, and the hyperparameter point with large variance is called `exploration`. Acquisition function is defined to balance the expected mean and variance. The currently selected hyperparameter point is viewed as the optimal position with maximum probability.

According to the above two search schemes, we carry out some experiments based on fixed scheme and two search schemes on 8 open source datasets. As the experimental scheme in [1], we search for 4 hyperparameters, the search space and The experimental results are as follows:

a fixed set of parameter experiments and two search schemes on 8 open source data sets. With reference to the experimental scheme of [1], we search for 4 hyperparameters, the search space and the experimental results are as follows:


- Fixed scheme.

```
lr=0.003，l2 decay=1e-4，label smoothing=False，mixup=False
```

- Search space of the hyperparameters.

```
lr: [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]

l2 decay: [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]

label smoothing: [False, True]

mixup: [False, True]
```

It takes 196 times for grid search, and takes 10 times less for Bayesian search. The baseline is trained by using ImageNet1k pretrained model based on ResNet50_vd and fixed scheme. The follow shows the experiments.


| Dataset             | Fix scheme | Grid search | Grid search time | Bayesian search | Bayesian search time|
| ------------------ | -------- | -------- | -------- | -------- | ---------- |
| Oxford-IIIT-Pets   | 93.64%   | 94.55%   | 196 | 94.04%     | 20         |
| Oxford-102-Flowers | 96.08%   | 97.69%   | 196 |  97.49%     | 20         |
| Food101            | 87.07%   | 87.52%   | 196 |  87.33%     | 23         |
| SUN397             | 63.27%   | 64.84%   | 196 |  64.55%     | 20         |
| Caltech101         | 91.71%   | 92.54%   | 196 |  92.16%     | 14         |
| DTD                | 76.87%   | 77.53%   | 196 |  77.47%     | 13         |
| Stanford Cars      | 85.14%   | 92.72%   | 196 |  92.72%     | 25         |
| FGVC Aircraft      | 80.32%   | 88.45%   | 196 |  88.36%     | 20         |


- The above experiments verify that Bayesian search only reduces the accuracy by 0% to 0.4% under the condition of reducing the number of searches by about 10 times compared to grid search.
- The search space can be expaned easily using Bayesian search.

<a name='2'></a>
## Large-scale image classification

In practical applications, due to the lack of training data, the classification model trained on the ImageNet1k data set is often used as the pretrained model for other image classification tasks. In order to further help solve practical problems, based on ResNet50_vd, Baidu open sourced a self-developed large-scale classification pretrained model, in which the training data contains 100,000 categories and 43 million pictures. The pretrained model can be downloaded as follows：[**download link**](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_10w_pretrained.pdparams)

We conducted transfer learning experiments on 6 self-collected datasets,

using a set of fixed parameters and a grid search method, in which the number of training rounds was set to 20epochs, the ResNet50_vd model was selected, and the ImageNet pre-training accuracy was 79.12%. The comparison results of the experimental data set parameters and model accuracy are as follows:


Fixed scheme：

```
lr=0.001，l2 decay=1e-4，label smoothing=False，mixup=False
```

| Dataset          | Statstics                                  | **Pretrained moel on ImageNet <br />Top-1(fixed)/Top-1(search)** | **Pretrained moel on large-scale dataset<br />Top-1(fixed)/Top-1(search)** |
| --------------- | ----------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| Flowers         | class:102<br />train:5789<br />valid:2396 | 0.7779/0.9883                                            | 0.9892/0.9954                                             |
| Hand-painted stick figures       | Class:18<br />train:1007<br />valid:432   | 0.8795/0.9196                                            | 0.9107/0.9219                                             |
| Leaves     | class:6<br />train:5256<br />valid:2278   | 0.8212/0.8482                                            | 0.8385/0.8659                                             |
| Container vehicle       | Class:115<br />train:4879<br />valid:2094 | 0.6230/0.9556                                            | 0.9524/0.9702                                             |
| Chair         | class:5<br />train:169<br />valid:78      | 0.8557/0.9688                                            | 0.9077/0.9792                                             |
| Geology         | class:4<br />train:671<br />valid:296     | 0.5719/0.8094                                            | 0.6781/0.8219                                             |

- The above experiments verified that for fixed parameters, compared with the pretrained model on ImageNet, using the large-scale classification model as a pretrained model can help us improve the model performance on a new dataset in most cases. Parameter search can be further helpful to the model performance.

<a name='3'></a>
## Reference

[1] Kornblith, Simon, Jonathon Shlens, and Quoc V. Le. "Do better imagenet models transfer better?." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2019.

[2] Kolesnikov, Alexander, et al. "Large Scale Learning of General Visual Representations for Transfer." *arXiv preprint arXiv:1912.11370* (2019).
