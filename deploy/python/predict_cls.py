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

import cv2
import numpy as np

from utils import logger
from utils import config
from utils.predictor import Predictor
from utils.get_image_list import get_image_list
from preprocess import create_operators
from postprocess import build_postprocess


class ClsPredictor(object):
    def __init__(self, config):
        super().__init__()
        self.predictor = Predictor(config["Global"])
        self.preprocess_ops = create_operators(config["PreProcess"][
            "transform_ops"])
        self.postprocess = build_postprocess(config["PostProcess"])


def main(config):
    cls_predictor = ClsPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    assert config["Global"]["batch_size"] == 1
    for idx, image_file in enumerate(image_list):
        batch_input = []
        img = cv2.imread(image_file)[:, :, ::-1]
        for ops in cls_predictor.preprocess_ops:
            img = ops(img)
        batch_input.append(img)
        output = cls_predictor.predictor.predict(np.array(batch_input))
        output = cls_predictor.postprocess(output)
        print(output)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
