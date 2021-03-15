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

from tools.infer.utils import parse_args, get_image_list, preprocess, np_to_b64
from ppcls.utils import logger
import numpy as np
import cv2
import time
import requests
import json
import base64


def main(args):
    image_path_list = get_image_list(args.image_file)
    headers = {"Content-type": "application/json"}

    cnt = 0
    predict_time = 0
    all_score = 0.0
    start_time = time.time()

    batch_input_list = []
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
            img = img[:, :, ::-1]
            data = preprocess(img, args)
            batch_input_list.append(data)
            img_name = img_path.split('/')[-1]
            img_name_list.append(img_name)
            cnt += 1
        if cnt % args.batch_size == 0 or (idx + 1) == len(image_path_list):
            batch_input = np.array(batch_input_list)
            b64str, revert_shape = np_to_b64(batch_input)
            data = {
                "images": b64str,
                "revert_params": {
                    "shape": revert_shape,
                    "dtype": str(batch_input.dtype)
                },
                "top_k": args.top_k
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
                batch_result_list = results["prediction"]
                elapse = results["elapse"]

                cnt += len(batch_result_list)
                predict_time += elapse

                for number, result_list in enumerate(batch_result_list):
                    all_score += result_list[0]["score"]
                    result_str = ", ".join([
                        "{}: {:.2f}".format(r["cls_id"], r["score"])
                        for r in result_list
                    ])
                    logger.info("File:{}, The top-{} result(s): {}".format(
                        img_name_list[number], args.top_k, result_str))

            finally:
                batch_input_list = []
                img_name_list = []

    total_time = time.time() - start_time
    logger.info("The average time of prediction cost: {:.3f} s/image".format(
        predict_time / cnt))
    logger.info("The average time cost: {:.3f} s/image".format(total_time /
                                                               cnt))
    logger.info("The average top-1 score: {:.3f}".format(all_score / cnt))


if __name__ == '__main__':
    args = parse_args()
    main(args)
