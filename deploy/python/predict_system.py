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
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import copy
import cv2
import numpy as np

from python.predict_rec import RecPredictor
from python.predict_det import DetPredictor

from utils import logger
from utils import config
from utils.get_image_list import get_image_list


class SystemPredictor(object):
    def __init__(self, config):
        self.rec_predictor = RecPredictor(config)
        self.det_predictor = DetPredictor(config)

    def predict(self, img):
        output = []
        results = self.det_predictor.predict(img)
        for result in results:
            print(result)
            xmin, xmax, ymin, ymax = result["bbox"].astype("int")
            crop_img = img[xmin:xmax, ymin:ymax, :].copy()
            rec_results = self.rec_predictor.predict(crop_img)
            result["featrue"] = rec_results
            output.append(result)
        return output


def main(config):
    system_predictor = SystemPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    assert config["Global"]["batch_size"] == 1
    for idx, image_file in enumerate(image_list):
        img = cv2.imread(image_file)[:, :, ::-1]
        output = system_predictor.predict(img)
        print(output)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
