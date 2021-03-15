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
import shutil
import os
import sys

import paddle
import paddle.nn.functional as F

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from ppcls.utils.save_load import load_dygraph_pretrain
from ppcls.utils import logger
from ppcls.modeling import architectures
from utils import parse_args, get_image_list, preprocess, postprocess


def save_prelabel_results(class_id, input_file_path, output_dir):
    output_dir = os.path.join(output_dir, str(class_id))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(input_file_path, output_dir)


def main():
    args = parse_args()
    # assign the place
    place = paddle.set_device('gpu' if args.use_gpu else 'cpu')

    net = architectures.__dict__[args.model](class_dim=args.class_num)
    load_dygraph_pretrain(net, args.pretrained_model, args.load_static_weights)
    image_list = get_image_list(args.image_file)
    batch_input_list = []
    img_path_list = []
    cnt = 0
    for idx, img_path in enumerate(image_list):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(
                "Image file failed to read and has been skipped. The path: {}".
                format(img_path))
            continue
        else:
            img = img[:, :, ::-1]
            data = preprocess(img, args)
            batch_input_list.append(data)
            img_path_list.append(img_path)
            cnt += 1

        if cnt % args.batch_size == 0 or (idx + 1) == len(image_list):
            batch_tensor = paddle.to_tensor(batch_input_list)
            net.eval()
            batch_outputs = net(batch_tensor)
            if args.model == "GoogLeNet":
                batch_outputs = batch_outputs[0]
            batch_outputs = F.softmax(batch_outputs)
            batch_outputs = batch_outputs.numpy()
            batch_result_list = postprocess(batch_outputs, args.top_k)

            for number, result_list in enumerate(batch_result_list):
                filename = img_path_list[number].split("/")[-1]
                result_str = ", ".join([
                    "{}: {:.2f}".format(r["cls_id"], r["score"])
                    for r in result_list
                ])
                print("File:{}, The top-{} result(s):{}".format(
                    filename, args.top_k, result_str))
                if args.pre_label_image:
                    save_prelabel_results(result_list[0]["cls_id"],
                                          img_path_list[number],
                                          args.pre_label_out_idr)
            batch_input_list = []
            img_path_list = []


if __name__ == "__main__":
    main()
