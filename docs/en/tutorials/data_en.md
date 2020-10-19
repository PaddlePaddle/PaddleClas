# Data

---

## Introducation
This document introduces the preparation of ImageNet1k and flowers102

## Dataset

Dataset | train dataset size | valid dataset size | category |
:------:|:---------------:|:---------------------:|:--------:|
[flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)|1k | 6k | 102 |
[ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 |

* Data format

Please follow the steps mentioned below to organize data, include train_list.txt and val_list.txt

```shell
# delimiter: "space"

ILSVRC2012_val_00000001.JPEG 65
...

```
### ImageNet1k
After downloading data, please organize the data dir as below

```bash
PaddleClas/dataset/imagenet/
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
### Flowers102 Dataset

Download [Data](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) then decompress:

```shell
jpg/
setid.mat
imagelabels.mat
```

Please put all the files under ```PaddleClas/dataset/flowers102```

generate generate_flowers102_list.py and train_list.txt和val_list.txt

```bash
python generate_flowers102_list.py jpg train > train_list.txt
python generate_flowers102_list.py jpg valid > val_list.txt

```

Please organize data dir as below

```bash
PaddleClas/dataset/flowers102/
|_ jpg/
|  |_ image_03601.jpg
|  |_ ...
|  |_ image_02355.jpg
|_ train_list.txt
|_ val_list.txt
```
