import os
import json
import bcolz
import numpy as np
from PIL import Image
import cv2
import paddle
import paddle.vision.datasets as datasets
from paddle.vision import transforms
from paddle.vision.transforms import functional as F
from paddle.io import Dataset
from .common_dataset import create_operators
from ppcls.data.preprocess import transform as transform_func

# code is based on AdaFace: https://github.com/mk-minchul/AdaFace


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif F._is_numpy_image(img):
        return img.shape[:2][::-1]
    elif F._is_tensor_image(img):
        return img.shape[1:][::-1]  # chw
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class AdaFaceDataset(Dataset):
    def __init__(
            self,
            root_dir,
            label_path,
            transform=None,
            low_res_augmentation_prob=0.0,
            crop_augmentation_prob=0.0,
            photometric_augmentation_prob=0.0, ):
        self.root_dir = root_dir
        self.low_res_augmentation_prob = low_res_augmentation_prob
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.random_resized_crop = transforms.RandomResizedCrop(
            size=(112, 112),
            scale=(0.2, 1.0),
            ratio=(0.75, 1.3333333333333333))
        self.photometric = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0)
        self.transform = create_operators(transform)

        self.tot_rot_try = 0
        self.rot_success = 0
        with open(label_path) as fd:
            lines = fd.readlines()
        self.samples = []
        for l in lines:
            l = l.strip().split()
            self.samples.append([os.path.join(root_dir, l[0]), int(l[1])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        [path, target] = self.samples[index]
        with open(path, 'rb') as f:
            img = Image.open(f)
            sample = img.convert('RGB')

        # if 'WebFace' in self.root:
        #     # swap rgb to bgr since image is in rgb for webface
        #     sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

        sample, _ = self.augment(sample)
        if self.transform is not None:
            sample = transform_func(sample, self.transform)

        return sample, target

    def augment(self, sample):

        # crop with zero padding augmentation
        if np.random.random() < self.crop_augmentation_prob:
            # RandomResizedCrop augmentation
            new = np.zeros_like(np.array(sample))
            #  orig_W, orig_H = F._get_image_size(sample)
            orig_W, orig_H = _get_image_size(sample)
            i, j, h, w = self.random_resized_crop._get_param(sample)
            cropped = F.crop(sample, i, j, h, w)
            new[i:i + h, j:j + w, :] = np.array(cropped)
            sample = Image.fromarray(new.astype(np.uint8))
            crop_ratio = min(h, w) / max(orig_H, orig_W)
        else:
            crop_ratio = 1.0

        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))
        else:
            resize_ratio = 1

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            sample = self.photometric(sample)
        information_score = resize_ratio * crop_ratio
        return sample, information_score


def low_res_augmentation(img):
    # resize the image to a small size and enlarge it back
    img_shape = img.shape
    side_ratio = np.random.uniform(0.2, 1.0)
    small_side = int(side_ratio * img_shape[0])
    interpolation = np.random.choice([
        cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4
    ])
    small_img = cv2.resize(
        img, (small_side, small_side), interpolation=interpolation)
    interpolation = np.random.choice([
        cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4
    ])
    aug_img = cv2.resize(
        small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)

    return aug_img, side_ratio


class FiveValidationDataset(Dataset):
    def __init__(self, val_data_path, concat_mem_file_name):
        '''
        concatenates all validation datasets from emore
        val_data_dict = {
        'agedb_30': (agedb_30, agedb_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
        }
        agedb_30: 0
        cfp_fp: 1
        lfw: 2
        cplfw: 3
        calfw: 4
        '''
        val_data = get_val_data(val_data_path)
        age_30, cfp_fp, lfw, age_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame = val_data
        val_data_dict = {
            'agedb_30': (age_30, age_30_issame),
            "cfp_fp": (cfp_fp, cfp_fp_issame),
            "lfw": (lfw, lfw_issame),
            "cplfw": (cplfw, cplfw_issame),
            "calfw": (calfw, calfw_issame),
        }
        self.dataname_to_idx = {
            "agedb_30": 0,
            "cfp_fp": 1,
            "lfw": 2,
            "cplfw": 3,
            "calfw": 4
        }

        self.val_data_dict = val_data_dict
        # concat all dataset
        all_imgs = []
        all_issame = []
        all_dataname = []
        key_orders = []
        for key, (imgs, issame) in val_data_dict.items():
            all_imgs.append(imgs)
            dup_issame = [
            ]  # hacky way to make the issame length same as imgs. [1, 1, 0, 0, ...]
            for same in issame:
                dup_issame.append(same)
                dup_issame.append(same)
            all_issame.append(dup_issame)
            all_dataname.append([self.dataname_to_idx[key]] * len(imgs))
            key_orders.append(key)
        assert key_orders == ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']

        if isinstance(all_imgs[0], np.memmap):
            self.all_imgs = read_memmap(concat_mem_file_name)
        else:
            self.all_imgs = np.concatenate(all_imgs)

        self.all_issame = np.concatenate(all_issame)
        self.all_dataname = np.concatenate(all_dataname)

    def __getitem__(self, index):
        x_np = self.all_imgs[index].copy()
        x = paddle.to_tensor(x_np)
        y = self.all_issame[index]
        dataname = self.all_dataname[index]
        return x, y, dataname, index

    def __len__(self):
        return len(self.all_imgs)


def read_memmap(mem_file_name):
    # r+ mode: Open existing file for reading and writing
    with open(mem_file_name + '.conf', 'r') as file:
        memmap_configs = json.load(file)
        return np.memmap(mem_file_name, mode='r+', \
                        shape=tuple(memmap_configs['shape']), \
                        dtype=memmap_configs['dtype'])


def get_val_pair(path, name, use_memfile=True):
    if use_memfile:
        mem_file_dir = os.path.join(path, name, 'memfile')
        mem_file_name = os.path.join(mem_file_dir, 'mem_file.dat')
        if os.path.isdir(mem_file_dir):
            print('laoding validation data memfile')
            np_array = read_memmap(mem_file_name)
        else:
            os.makedirs(mem_file_dir)
            carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
            np_array = np.array(carray)
            #  mem_array = make_memmap(mem_file_name, np_array)
            #  del np_array, mem_array
            del np_array
            np_array = read_memmap(mem_file_name)
    else:
        np_array = bcolz.carray(rootdir=os.path.join(path, name), mode='r')

    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))
    return np_array, issame


def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame