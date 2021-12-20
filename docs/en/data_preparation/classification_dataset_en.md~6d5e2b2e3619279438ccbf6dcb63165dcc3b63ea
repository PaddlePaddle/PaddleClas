# Image Classification Datasets

This document elaborates on the dataset format adopted by PaddleClas for image classification tasks, as well as other common datasets in this field.

------

## Catalogue

- [1.Dataset Format](#1)
- [2.Common Datasets for Image Classification](#2)
  - [2.1 ImageNet1k](#2.1)
  - [2.2 Flowers102](#2.2)
  - [2.3 CIFAR10 / CIFAR100](#2.3)
  - [2.4 MNIST](#2.4)
  - [2.5 NUS-WIDE](#2.5)


<a name="1"></a>
## 1.Dataset Format

PaddleClas adopts `txt` files to assign the training and test sets. Taking the `ImageNet1k` dataset as an example, where `train_list.txt` and `val_list.txt` have the following formats:

```
# Separate the image path and annotation with "space" for each line

# train_list.txt has the following format
train/n01440764/n01440764_10026.JPEG 0
...

# val_list.txt has the following format
val/ILSVRC2012_val_00000001.JPEG 65
...
```


<a name="2"></a>
## 2.Common Datasets for Image Classification

Here we present a compilation of commonly used image classification datasets, which is continuously updated and expects your supplement.

<a name="2.1"></a>
### 2.1 ImageNet1k

[ImageNet](https://image-net.org/) is a large visual database for visual target recognition research with over 14 million manually labeled images. ImageNet-1k is a subset of the ImageNet dataset, which contains 1000 categories with 1281167 images for the training set and 50000 for the validation set. Since 2010, ImageNet began to hold an annual image classification competition, namely, the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) with ImageNet-1k as its specified dataset. To date, ImageNet-1k has become one of the most significant contributors to the development of computer vision, based on which numerous initial models of downstream computer vision tasks are trained.

| Dataset                                                      | Size of Training Set | Size of Test Set | Number of Category | Note |
| ------------------------------------------------------------ | -------------------- | ---------------- | ------------------ | ---- |
| [ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/) | 1.2M                 | 50k              | 1000               |      |

After downloading the data from official sources, organize it in the following format to train with the ImageNet1k dataset in PaddleClas.

```
PaddleClas/dataset/ILSVRC2012/
|_ train/
|  |_ n01440764
|  |  |_ n01440764_10026.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ n15075141
|     |_ ...
|     |_ n15075141_9993.JPEG
|_ val/
|  |_ ILSVRC2012_val_00000001.JPEG
|  |_ ...
|  |_ ILSVRC2012_val_00050000.JPEG
|_ train_list.txt
|_ val_list.txt
```


<a name="2.2"></a>
### 2.2 Flowers102

| Dataset                                                      | Size of Training Set | Size of Test Set | Number of Category | Note |
| ------------------------------------------------------------ | -------------------- | ---------------- | ------------------ | ---- |
| [flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | 1k                   | 6k               | 102                |      |

Unzip the downloaded data to see the following directory.

```
jpg/
setid.mat
imagelabels.mat
```

Place the files above under `PaddleClas/dataset/flowers102/` .

Run `generate_flowers102_list.py` to generate `train_list.txt` and `val_list.txt`:

```
python generate_flowers102_list.py jpg train > train_list.txt
python generate_flowers102_list.py jpg valid > val_list.txt
```

Structure the data as follows：

```
PaddleClas/dataset/flowers102/
|_ jpg/
|  |_ image_03601.jpg
|  |_ ...
|  |_ image_02355.jpg
|_ train_list.txt
|_ val_list.txt
```


<a name="2.3"></a>
### 2.3 CIFAR10 / CIFAR100

The CIFAR-10 dataset comprises 60,000 color images of 10 classes with 32x32 image resolution, each with 6,000 images including 5,000 images in the training set and 1,000 images in the validation set. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The CIFAR-100 dataset is an extension of CIFAR-10 and consists of 60,000 color images of 100 classes with 32x32 image resolution, each with 600 images including 500 images in the training set and 100 images in the validation set.

Website：http://www.cs.toronto.edu/~kriz/cifar.html


<a name="2.4"></a>
### 2.4 MNIST

MMNIST is a renowned dataset for handwritten digit recognition and is used as an introductory sample for deep learning in many sources. It contains 60,000 images, 50,000 for the training set and 10,000 for the validation set, with a size of 28 * 28.

Website：http://yann.lecun.com/exdb/mnist/


<a name="2.5"></a>
### 2.5 NUS-WIDE

NUS-WIDE is a multi-category dataset. It contains 269,648 images and 81 categories with each image being labeled as one or more of the 81 categories.

Website：https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
