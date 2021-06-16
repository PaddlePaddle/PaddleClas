# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import time
import requests
import json
import base64
import argparse

import numpy as np
import cv2

from utils import logger
from utils.get_image_list import get_image_list
from utils import config
from utils.encode_decode import np_to_b64
from python.preprocess import create_operators

preprocess_config = [{
    'ResizeImage': {
        'resize_short': 256
    }
}, {
    'CropImage': {
        'size': 224
    }
}, {
    'NormalizeImage': {
        'scale': 0.00392157,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'order': ''
    }
}, {
    'ToCHWImage': None
}]


def main(args):
    image_path_list = get_image_list(args.image_file)
    headers = {"Content-type": "application/json"}
    preprocess_ops = create_operators(preprocess_config)

    cnt = 0
    predict_time = 0
    all_score = 0.0
    start_time = time.time()

    img_data_list = []
    img_name_list = []
    cnt = 0
    for idx, img_path in enumerate(image_path_list):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(
                "Image file failed to read and has been skipped. The path: {}".
                format(img_path))
            continue
        else:
            for ops in preprocess_ops:
                img = ops(img)
            img = np.array(img)
            img_data_list.append(img)

            img_name = img_path.split('/')[-1]
            img_name_list.append(img_name)
            cnt += 1
        if cnt % args.batch_size == 0 or (idx + 1) == len(image_path_list):
            inputs = np.array(img_data_list)
            b64str, revert_shape = np_to_b64(inputs)
            data = {
                "images": b64str,
                "revert_params": {
                    "shape": revert_shape,
                    "dtype": str(inputs.dtype)
                }
            }
            try:
                r = requests.post(
                    url=args.server_url,
                    headers=headers,
                    data=json.dumps(data))
                r.raise_for_status
                if r.json()["status"] != "000":
                    msg = r.json()["msg"]
                    raise Exception(msg)
            except Exception as e:
                logger.error("{}, in file(s): {} etc.".format(e, img_name_list[
                    0]))
                continue
            else:
                results = r.json()["results"]
                preds = results["prediction"]
                elapse = results["elapse"]

                cnt += len(preds)
                predict_time += elapse

                for number, result_list in enumerate(preds):
                    all_score += result_list["scores"][0]
                    result_str = ""
                    for i in range(len(result_list["class_ids"])):
                        result_str += "{}: {:.2f}\t".format(
                            result_list["class_ids"][i],
                            result_list["scores"][i])

                    logger.info("File:{}, The result(s): {}".format(
                        img_name_list[number], result_str))

            finally:
                img_data_list = []
                img_name_list = []

    total_time = time.time() - start_time
    logger.info("The average time of prediction cost: {:.3f} s/image".format(
        predict_time / cnt))
    logger.info("The average time cost: {:.3f} s/image".format(total_time /
                                                               cnt))
    logger.info("The average top-1 score: {:.3f}".format(all_score / cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str)
    parser.add_argument("--image_file", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    main(args)
