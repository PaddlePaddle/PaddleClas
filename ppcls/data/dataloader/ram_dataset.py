import json
import os
import random

from paddle.io import Dataset
import paddle
from paddle.vision.transforms import Resize, Compose, Resize, ToTensor, Normalize

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import os, glob
import re
import numpy as np


def convert_to_rgb(image):
    return image.convert("RGB")


def get_transform(image_size=384):
    return Compose([
        convert_to_rgb, Resize((image_size, image_size)), ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(), )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption, )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


class RAM_pretrain_dataset(Dataset):
    def __init__(self, ann_file, width=384, class_num=4585, root=''):

        self.ann = []
        for f in ann_file:
            print('loading ' + f)
            ann = json.load(open(f, 'r'))
            self.ann += ann
        self.width = width
        self.resize_op = Resize(224)
        self.transform = get_transform(width)
        self.class_num = class_num
        self.root = root

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path_use = os.path.join(self.root, ann['image_path'])

        try:
            image = Image.open(image_path_use).convert('RGB')
            image = self.transform(image)
        except:
            image = paddle.ones([3, self.width, self.width])

        num = ann['union_label_id']
        image_tag = np.zeros([self.class_num])
        image_tag[num] = 1
        image_tag = paddle.to_tensor(image_tag, dtype=paddle.int32)

        caption_index = np.random.randint(0, len(ann['caption']))

        caption = pre_caption(ann['caption'][caption_index], 30)

        num = ann['parse_label_id'][caption_index]
        parse_tag = np.zeros([self.class_num])
        parse_tag[num] = 1
        parse_tag = paddle.to_tensor(parse_tag, dtype=paddle.int32)
        image224 = self.resize_op(image)

        return (image, caption, image_tag, parse_tag, image224)


class RAM_finetune_dataset(Dataset):
    def __init__(self, ann_file, width=384, class_num=4585, root=''):

        self.ann = []
        for f in ann_file:
            print('loading ' + f)
            ann = json.load(open(f, 'r'))
            self.ann += ann
        self.width = width
        self.resize_op = Resize(224)
        self.transform = get_transform(width)
        self.transform_224 = get_transform(224)
        self.class_num = class_num
        self.root = root

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path_use = os.path.join(self.root, ann['image_path'])
        image = Image.open(image_path_use).convert('RGB')
        image = self.transform(image)

        image_224 = Image.open(image_path_use).convert('RGB')
        image_224 = self.transform_224(image_224)

        num = ann['union_label_id']
        image_tag = np.zeros([self.class_num])
        image_tag[num] = 1
        image_tag = paddle.to_tensor(image_tag, dtype=paddle.int32)

        caption_index = np.random.randint(0, len(ann['caption']))

        caption = pre_caption(ann['caption'][caption_index], 30)

        num = ann['parse_label_id'][caption_index]
        parse_tag = np.zeros([self.class_num])
        parse_tag[num] = 1
        parse_tag = paddle.to_tensor(parse_tag, dtype=paddle.int32)
        image224 = self.resize_op(image)

        return (image, caption, image_tag, parse_tag, image224)
