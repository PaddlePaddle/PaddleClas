from functools import reduce
import numpy as np

from ...utils import logger
from ..base_processor import BaseProcessor


class DetPostPro(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.threshold = config["threshold"]
        self.label_list = config["label_list"]
        self.max_det_results = config["max_det_results"]
        if self.input_keys is None:
            self.input_keys = ["pred"]
        if self.output_keys is None:
            self.output_keys = ["det_result"]

    def process(self, input_data):
        np_boxes = input_data[self.input_keys[0]]["boxes"]
        if reduce(lambda x, y: x * y, np_boxes.shape) >= 6:
            keep_indexes = np_boxes[:, 1].argsort()[::-1][:
                                                          self.max_det_results]

            all_results = []
            for idx in keep_indexes:
                single_res = np_boxes[idx]
                class_id = int(single_res[0])
                score = single_res[1]
                bbox = single_res[2:]
                if score < self.threshold:
                    continue
                label_name = self.label_list[class_id]
                all_results.append({
                    "class_id": class_id,
                    "score": score,
                    "bbox": bbox,
                    "label_name": label_name
                })
            input_data[self.output_keys[0]] = all_results
            return input_data

        logger.warning('[Detector] No object detected.')
        input_data[self.output_keys[0]] = []
        return input_data
