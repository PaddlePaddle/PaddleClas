# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os ,glob
import random
import re
import numpy as np

from paddle.io import Dataset
import paddle
from paddle.vision.transforms import Resize, Compose, Resize, ToTensor, Normalize
from .common_dataset import create_operators
from ppcls.data.preprocess import transform

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None



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


class RAMPretrainDataset(Dataset):
    def __init__(self, ann_file, width=384, class_num=4585, root='',transform_ops_ram=None, transform_ops_clip=None):

        self.ann = []
        for f in ann_file:
            print('loading ' + f)
            ann = json.load(open(f, 'r'))
            self.ann += ann
        self.width = width
        self.transform_clip = create_operators(transform_ops_clip)
        self.transform = create_operators(transform_ops_ram)
        self.class_num = class_num
        self.root = root

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path_use = os.path.join(self.root, ann['image_path'])

        try:
            image = Image.open(image_path_use).convert('RGB')
            image_ram = transform(image, self.transform)
            image224 = transform(image, self.transform_clip)
        except:
            image224 = paddle.ones([3, 224, 224])
            image_ram = paddle.ones([3, self.width, self.width])

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
        

        return (image_ram, caption, image_tag, parse_tag, image224)


class RAMFinetuneDataset(Dataset):
    def __init__(self, ann_file, width=384, class_num=4585, root='', transform_ops_ram=None, transform_ops_clip=None):

        self.ann = []
        for f in ann_file:
            print('loading ' + f)
            ann = json.load(open(f, 'r'))
            self.ann += ann
        self.width = width
        self.transform_clip = create_operators(transform_ops_clip)
        self.transform = create_operators(transform_ops_ram)
        self.class_num = class_num
        self.root = root

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path_use = os.path.join(self.root, ann['image_path'])
        image = Image.open(image_path_use).convert('RGB')
        image_ram = transform(image, self.transform)

        image_224 = Image.open(image_path_use).convert('RGB')
        image_224 = transform(image, self.transform_clip)

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

        return (image_ram, caption, image_tag, parse_tag, image_224)
