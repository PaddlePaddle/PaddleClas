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
        self.benchmark = config["Global"].get("benchmark", False)
        if self.benchmark:
            import auto_log
            import os
            pid = os.getpid()
            size = config["PreProcess"]["transform_ops"][1]["CropImage"][
                "size"]
            if config["Global"].get("use_int8", False):
                precision = "int8"
            elif config["Global"].get("use_fp16", False):
                precision = "fp16"
            else:
                precision = "fp32"
            self.auto_logger = auto_log.AutoLogger(
                model_name=config["Global"].get("model_name", "cls"),
                model_precision=precision,
                batch_size=config["Global"].get("batch_size", 1),
                data_shape=[3, size, size],
                save_path=config["Global"].get("save_log_path",
                                               "./auto_log.log"),
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=2)

    def predict(self, images):
        use_onnx = self.args.get("use_onnx", False)
        if not use_onnx:
            input_names = self.predictor.get_input_names()
            input_tensor = self.predictor.get_input_handle(input_names[0])

            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
        else:
            input_names = self.predictor.get_inputs()[0].name
            output_names = self.predictor.get_outputs()[0].name

        if self.benchmark:
            self.auto_logger.times.start()
        if not isinstance(images, (list, )):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)
        if self.benchmark:
            self.auto_logger.times.stamp()

        if not use_onnx:
            input_tensor.copy_from_cpu(image)
            self.predictor.run()
            batch_output = output_tensor.copy_to_cpu()
        else:
            batch_output = self.predictor.run(
                output_names=[output_names],
                input_feed={input_names: image})[0]

        if self.benchmark:
            self.auto_logger.times.stamp()
        if self.postprocess is not None:
            batch_output = self.postprocess(batch_output)
        if self.benchmark:
            self.auto_logger.times.end(stamp=True)
        return batch_output


def main(config):
    cls_predictor = ClsPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    batch_imgs = []
    batch_names = []
    cnt = 0
    for idx, img_path in enumerate(image_list):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(
                "Image file failed to read and has been skipped. The path: {}".
                format(img_path))
        else:
            img = img[:, :, ::-1]
            batch_imgs.append(img)
            img_name = os.path.basename(img_path)
            batch_names.append(img_name)
            cnt += 1

        if cnt % config["Global"]["batch_size"] == 0 or (idx + 1
                                                         ) == len(image_list):
            if len(batch_imgs) == 0:
                continue
            batch_results = cls_predictor.predict(batch_imgs)
            for number, result_dict in enumerate(batch_results):
                if "PersonAttribute" in config[
                        "PostProcess"] or "VehicleAttribute" in config[
                            "PostProcess"]:
                    filename = batch_names[number]
                    print("{}:\t {}".format(filename, result_dict))
                else:
                    filename = batch_names[number]
                    clas_ids = result_dict["class_ids"]
                    scores_str = "[{}]".format(", ".join("{:.2f}".format(
                        r) for r in result_dict["scores"]))
                    label_names = result_dict["label_names"]
                    print(
                        "{}:\tclass id(s): {}, score(s): {}, label_name(s): {}".
                        format(filename, clas_ids, scores_str, label_names))
            batch_imgs = []
            batch_names = []
    if cls_predictor.benchmark:
        cls_predictor.auto_logger.report()
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
