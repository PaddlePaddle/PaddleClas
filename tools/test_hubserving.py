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

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from ppcls.utils import logger
import cv2
import time
import requests
import json
import base64
import imghdr


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')


def main(url, image_path, top_k=1):
    image_file_list = get_image_file_list(image_path)
    headers = {"Content-type": "application/json"}
    cnt = 0
    total_time = 0
    all_acc = 0.0

    for image_file in image_file_list:
        file_str = image_file.split('/')[-1]
        img = open(image_file, 'rb').read()
        if img is None:
            logger.error("Loading image:{} failed".format(image_file))
            continue
        data = {'images': [cv2_to_base64(img)], 'top_k': top_k}

        try:
            r = requests.post(url=url, headers=headers, data=json.dumps(data))
            r.raise_for_status()
        except Exception as e:
            logger.error("File:{}, {}".format(file_str, e))
            continue
        if r.json()['status'] != '000':
            logger.error(
                "File:{}, The parameters returned by the server are: {}".
                format(file_str, r.json()['msg']))
            continue
        res = r.json()["results"][0]
        classes = res[0]
        scores = res[1]
        elapse = res[2]
        all_acc += scores[0]
        total_time += elapse
        cnt += 1

        scores = map(lambda x: round(x, 5), scores)
        results = dict(zip(classes, scores))

        message = "No.{}, File:{}, The top-{} result(s):{}, Time cost:{:.3f}".format(
            cnt, file_str, top_k, results, elapse)
        logger.info(message)

    logger.info("The average time cost: {}".format(float(total_time) / cnt))
    logger.info("The average top-1 score: {}".format(float(all_acc) / cnt))


if __name__ == '__main__':
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        logger.info("Usage: %s server_url image_path" % sys.argv[0])
    else:
        server_url = sys.argv[1]
        image_path = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) == 4 else 1
        main(server_url, image_path, top_k)
