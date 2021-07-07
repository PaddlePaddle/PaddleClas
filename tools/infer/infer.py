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

import numpy as np
import argparse
import utils
import shutil
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from ppcls.utils.save_load import load_dygraph_pretrain
from ppcls.modeling import architectures

import paddle
from paddle.distributed import ParallelEnv
import paddle.nn.functional as F


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_file", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-p", "--pretrained_model", type=str)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
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


def create_operators():
    size = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    decode_op = utils.DecodeImage()
    resize_op = utils.ResizeImage(resize_short=256)
    crop_op = utils.CropImage(size=(size, size))
    normalize_op = utils.NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = utils.ToTensor()

    return [decode_op, resize_op, crop_op, normalize_op, totensor_op]


def preprocess(fname, ops):
    data = open(fname, 'rb').read()
    for op in ops:
        data = op(data)
    return data


def postprocess(outputs, topk=5):
    output = outputs[0]
    prob = np.array(output).flatten()
    index = prob.argsort(axis=0)[-topk:][::-1].astype('int32')
    return zip(index, prob[index])


def get_image_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp']
    if os.path.isfile(img_file) and img_file.split('.')[-1] in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            if single_file.split('.')[-1] in img_end:
                imgs_lists.append(os.path.join(img_file, single_file))
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


def save_prelabel_results(class_id, input_filepath, output_idr):
    output_dir = os.path.join(output_idr, str(class_id))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(input_filepath, output_dir)


def main():
    args = parse_args()
    operators = create_operators()
    # assign the place
    if args.use_gpu:
        gpu_id = ParallelEnv().dev_id
        place = paddle.CUDAPlace(gpu_id)
    else:
        place = paddle.CPUPlace()

    paddle.disable_static(place)

    net = architectures.__dict__[args.model](class_dim=args.class_num)
    load_dygraph_pretrain(net, args.pretrained_model, args.load_static_weights)
    image_list = get_image_list(args.image_file)
    for idx, filename in enumerate(image_list):
        data = preprocess(filename, operators)
        data = np.expand_dims(data, axis=0)
        data = paddle.to_tensor(data)
        net.eval()
        outputs = net(data)
        if args.model == "GoogLeNet":
            outputs = outputs[0]
        else:
            outputs = F.softmax(outputs)
        outputs = outputs.numpy()
        probs = postprocess(outputs)

        top1_class_id = 0
        rank = 1
        print("Current image file: {}".format(filename))
        for idx, prob in probs:
            print("\ttop{:d}, class id: {:d}, probability: {:.4f}".format(
                rank, idx, prob))
            if rank == 1:
                top1_class_id = idx
            rank += 1

        if args.pre_label_image:
            save_prelabel_results(top1_class_id, filename,
                                  args.pre_label_out_idr)

    return


if __name__ == "__main__":
    main()
