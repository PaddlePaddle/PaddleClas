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
import cv2
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
import paddle.nn.functional as F


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
    args = utils.parse_args()
    # assign the place
    place = paddle.set_device('gpu' if args.use_gpu else 'cpu')

    net = architectures.__dict__[args.model](class_dim=args.class_num)
    load_dygraph_pretrain(net, args.pretrained_model, args.load_static_weights)
    image_list = get_image_list(args.image_file)
    for idx, filename in enumerate(image_list):
        img = cv2.imread(filename)[:, :, ::-1]
        data = utils.preprocess(img, args)
        data = np.expand_dims(data, axis=0)
        data = paddle.to_tensor(data)
        net.eval()
        outputs = net(data)
        if args.model == "GoogLeNet":
            outputs = outputs[0]
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
