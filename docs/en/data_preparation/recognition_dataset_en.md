# Image Recognition Datasets

This document elaborates on the dataset format adopted by PaddleClas for image recognition tasks, as well as other common datasets in this field.

------

## Catalogue

- [1.Dataset Format](#1)
- [2.Common Datasets for Image Recognition](#2)
  - [2.1 General Datasets](#2.1)
  - [2.2 Vertical Class Datasets](#2.2)
    - [2.2.1 Animation Character Recognition](#2.2.1)
    - [2.2.2 Product Recognition](#2.2.2)
    - [2.2.3 Logo Recognition](#2.2.3)
    - [2.2.4 Vehicle Recognition](#2.2.4)


<a name="1"></a>
## 1.Dataset Format

The dataset for the vector search, unlike those for classification tasks, is divided into the following three parts:

- Train dataset: Used to train the model to learn the image features involved.
- Gallery dataset: Used to provide the gallery data in the vector search task. It can either be the same as the train or query datasets or different, and when it is the same as the train dataset, the category system of the query dataset and train dataset should be the same.
- Query dataset: Used to test the performance of the model. It usually extracts features from each query image of the dataset, followed by distance matching with those in the gallery dataset to get the recognition results, based on which the metrics of the whole query dataset are calculated.

The above three datasets all adopt  `txt` files for assignment. Taking the `CUB_200_2011` dataset as an example, the `train_list.txt` of the train dataset has the following format：

```
# Use "space" as the separator
...
train/99/Ovenbird_0136_92859.jpg 99 2
...
train/99/Ovenbird_0128_93366.jpg 99 6
...
```

The `test_list.txt` of the query dataset (both gallery dataset and query dataset in`CUB_200_2011`) has the following format：

```
# Use "space" as the separator
...
test/200/Common_Yellowthroat_0126_190407.jpg 200 1
...
test/200/Common_Yellowthroat_0114_190501.jpg 200 6
...
```

Each row of data is separated by "space", and the three columns of data stand for the path, label information, and unique id of training data.

**Note**：

1. When the gallery dataset and query dataset are the same, to remove the first retrieved data (the images themselves require no evaluation), each data should have its unique id (ensuring that each image has a different id, which can be represented by the row number) for subsequent evaluation of mAP, recall@1, and other metrics. The dataset of yaml configuration file is `VeriWild`.

2. When the gallery dataset and query dataset are different, there is no need to add a unique id. Both `query_list.txt` and `gallery_list.txt` contain two columns, which are the path and label information of the training data. The dataset of yaml configuration file is ` ImageNetDataset`.


<a name="2"></a>
## 2.Common Datasets for Image Recognition

Here we present a compilation of commonly used image recognition datasets, which is continuously updated and expects your supplement.

<a name="2.1"></a>
### 2.1 General Datasets

- SOP: The SOP dataset is a common product dataset in general recognition research and MetricLearning technology research, which contains 120,053 images of 22,634 products downloaded from eBay.com. There are 59,551 images of 11,318 in the training set and 60,502 images of 11,316 categories in the validation set.

  Website: https://cvgl.stanford.edu/projects/lifted_struct/

- Cars196: The Cars dataset contains 16,185 images of 196 categories of cars. The data is classified into 8144 training images and 8041 query images, with each category split roughly in a 50-50 ratio. The classification is normally based on the manufacturing, model and year of the car, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

  Website: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

- CUB_200_2011: The CUB_200_2011 dataset is a fine-grained dataset proposed by the California Institute of Technology (Caltech) in 2010 and is currently the benchmark image dataset for fine-grained classification recognition research. There are 11788 bird images in this dataset with 200 subclasses, including 5994 images in the train dataset and 5794 images in the query dataset. Each image provides label information, the bounding box of the bird, the key part information of the bird, and the attribute of the bird. The dataset is shown in the figure below.

- In-shop Clothes: In-shop Clothes is one of the 4 subsets of the DeepFashion dataset. It is a seller show image dataset with multi-angle images of each product id being collected in the same folder. The dataset contains 7982 items with 52712 images, each with 463 attributes, Bbox, landmarks, and store descriptions.

  Website： http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html


<a name="2.2"></a>
### 2.2 Vertical Class Datasets


<a name="2.2.1"></a>
#### 2.2.1 Animation Character Recognition

- iCartoonFace: iCartoonFace, developed by iQiyi (an online video platform), is the world's largest manual labeled detection and recognition dataset for cartoon characters, which contains more than 5013 cartoon characters and 389,678 high-quality live images. Compared with other datasets, it boasts features of large scale, high quality, rich diversity, and challenging difficulty, making it one of the most commonly used datasets to study cartoon character recognition.

- Website： http://challenge.ai.iqiyi.com/detail?raceId=5def69ace9fcf68aef76a75d

- Manga109: Manga109 is a dataset released in May 2020 for the study of cartoon character detection and recognition, which contains 21142 images and is officially banned from commercial use. Manga109-s, a subset of this dataset, is available for industrial use, mainly for tasks such as text detection, sketch line-based search, and character image generation.

  Website：http://www.manga109.org/en/

- IIT-CFW: The IIF-CFW dataset contains a total of 8928 labeled cartoon portraits of celebrity characters, covering 100 characters with varying numbers of portraits for each. In addition, it also provides 1000 real face photos (10 real portraits for 100 public figures). This dataset can be employed to study both animation character recognition and cross-modal search tasks.

  Website： http://cvit.iiit.ac.in/research/projects/cvit-projects/cartoonfaces


<a name="2.2.2"></a>
#### 2.2.2 Product Recognition

- AliProduct: The AliProduct dataset is the largest open source product dataset. As an SKU-level image classification dataset, it contains 50,000 categories and 3 million images, ranking the first in both aspects in the industry. This dataset covers a large number of household goods, food, etc. Due to its lack of manual annotation, the data is messy and unevenly distributed with many similar product images.

  Website: https://retailvisionworkshop.github.io/recognition_challenge_2020/

- Product-10k: Products-10k dataset has all its images from Jingdong Mall, covering 10,000 frequently purchased SKUs that are organized into a hierarchy. In total, there are nearly 190,000 images. In the real application scenario, the distribution of image volume is uneven. All images are manually checked/labeled by a team of production experts.

  Website：https://www.kaggle.com/c/products-10k/data?select=train.csv

- DeepFashion-Inshop: The same as the common datasets In-shop Clothes.


<a name="2.2.3"></a>
### 2.2.3 Logo Recognition

- Logo-2K+: Logo-2K+ is a dataset exclusively for logo image recognition, which contains 10 major categories, 2341 minor categories, and 167,140 images.

  Website： https://github.com/msn199959/Logo-2k-plus-Dataset

- Tsinghua-Tencent 100K: This dataset is a large traffic sign benchmark dataset based on 100,000 Tencent Street View panoramas. 30,000 traffic sign instances included, it provides 100,000 images covering a wide range of illumination, and weather conditions. Each traffic sign in the benchmark test is labeled with the category, bounding box and pixel mask. A total of 222 categories (0 background + 221 traffic signs) are incorporated.

  Website： https://cg.cs.tsinghua.edu.cn/traffic-sign/


<a name="2.2.4"></a>
### 2.2.4 Vehicle Recognition

- CompCars: The images, 136,726 images of the whole car and 27,618 partial ones, are mainly from network and surveillance data. The network data contains 163 vehicle manufacturers and 1,716 vehicle models and includes the bounding box, viewing angle, and 5 attributes (maximum speed, displacement, number of doors, number of seats, and vehicle type). And the surveillance data comprises 50,000 front view images.

  Website： http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

- BoxCars: The dataset contains a total of 21,250 vehicles, 63,750 images, 27 vehicle manufacturers, and 148 subcategories. All of them are derived from surveillance data.

  Website： https://github.com/JakubSochor/BoxCars

- PKU-VD Dataset: The dataset contains two large vehicle datasets (VD1 and VD2) that capture images from real-world unrestricted scenes in two cities. VD1 is obtained from high-resolution traffic cameras, while images in VD2 are acquired from surveillance videos. The authors have performed vehicle detection on the raw data to ensure that each image contains only one vehicle. Due to privacy constraints, all the license numbers have been obscured with black overlays. All images are captured from the front view, and diverse attribute annotations are provided for each image in the dataset, including identification numbers, accurate vehicle models, and colors. VD1 originally contained 1097649 images, 1232 vehicle models, and 11 vehicle colors, and remains 846358 images and 141756 vehicles after removing images with multiple vehicles inside and those taken from the rear of the vehicle. VD2 contains 807260 images, 79763 vehicles, 1112 vehicle models, and 11 vehicle colors.

  Website： https://pkuml.org/resources/pku-vds.html
