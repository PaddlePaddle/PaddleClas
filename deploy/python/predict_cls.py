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
from python.preprocess import create_operators
from python.postprocess import build_postprocess


class ClsPredictor(Predictor):
    def __init__(self, config):
        super().__init__(config["Global"])

        self.preprocess_ops = []
        self.postprocess = None
        if "PreProcess" in config:
            if "transform_ops" in config["PreProcess"]:
                self.preprocess_ops = create_operators(config["PreProcess"][
                    "transform_ops"])
        if "PostProcess" in config:
            self.postprocess = build_postprocess(config["PostProcess"])

        # for whole_chain project to test each repo of paddle
        self.benchmark = config.get(["benchmark"], False)
        if self.benchmark:
            import auto_log
            import os
            pid = os.getpid()
            self.auto_log = auto_log.AutoLogger(
                model_name='cls',
                model_precision='fp16'
                if config["Global"]["use_fp16"] else 'fp32',
                batch_size=1,
                data_shape=[3, 224, 224],
                save_path="../output/auto_log.lpg",
                inference_config=None,
                pids=pid,
                process_name=None,
                gpu_ids=None,
                time_keys=['preprocess_time', 'inference_time'],
                warmup=10)

    def predict(self, images):
        input_names = self.paddle_predictor.get_input_names()
        input_tensor = self.paddle_predictor.get_input_handle(input_names[0])

        output_names = self.paddle_predictor.get_output_names()
        output_tensor = self.paddle_predictor.get_output_handle(output_names[
            0])

        if self.benchmark:
            self.auto_log.times.start()
        if not isinstance(images, (list, )):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)
        if self.benchmark:
            self.auto_log.times.stamp()

        input_tensor.copy_from_cpu(image)
        self.paddle_predictor.run()
        batch_output = output_tensor.copy_to_cpu()
        if self.benchmark:
            self.auto_log.times.stamp()
        return batch_output


def main(config):
    cls_predictor = ClsPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    assert config["Global"]["batch_size"] == 1
    for idx, image_file in enumerate(image_list):
        img = cv2.imread(image_file)[:, :, ::-1]
        output = cls_predictor.predict(img)
        output = cls_predictor.postprocess(output, [image_file])
        if cls_predictor.benchmark:
            cls_predictor.auto_log.times.end(stamp=True)
            cls_predictor.auto_log.report()
        print(output)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
