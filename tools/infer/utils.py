# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import numpy as np


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_file", type=str)
    parser.add_argument("--use_gpu", type=str2bool, default=True)

    # params for preprocess
    parser.add_argument("--resize_short", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--normalize", type=str2bool, default=True)

    # params for predict and predict_system
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--model_name", type=str)

    # params for infer
    parser.add_argument("--model", type=str)
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--class_num", type=int, default=1000)
    parser.add_argument(
        "--load_static_weights",
        type=str2bool,
        default=False,
        help='Whether to load the pretrained weights saved in static mode')

    # parameters for pre-label the images
    parser.add_argument(
        "--pre_label_image",
        type=str2bool,
        default=False,
        help="Whether to pre-label the images using the loaded weights")
    parser.add_argument("--pre_label_out_idr", type=str, default=None)

    return parser.parse_args()


def preprocess(img, args):
    resize_op = ResizeImage(resize_short=args.resize_short)
    img = resize_op(img)
    crop_op = CropImage(size=(args.resize, args.resize))
    img = crop_op(img)
    if args.normalize:
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        img_scale = 1.0 / 255.0
        normalize_op = NormalizeImage(
            scale=img_scale, mean=img_mean, std=img_std)
        img = normalize_op(img)
    tensor_op = ToTensor()
    img = tensor_op(img)
    return img


class ResizeImage(object):
    def __init__(self, resize_short=None):
        self.resize_short = resize_short

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        percent = float(self.resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
        return cv2.resize(img, (w, h))


class CropImage(object):
    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None):
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        return (img.astype('float32') * self.scale - self.mean) / self.std


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return img


class Base64ToCV2(object):
    def __init__(self):
        pass

    def __call__(self, b64str):
        import base64
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.fromstring(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)[:, :, ::-1]
        return data
