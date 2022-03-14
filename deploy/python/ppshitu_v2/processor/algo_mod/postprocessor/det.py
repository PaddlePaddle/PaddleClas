from functools import reduce
import numpy as np

from utils import logger
from ...base_processor import BaseProcessor


class DetPostPro(BaseProcessor):
    def __init__(self, config):
        self.threshold = config["threshold"]
        self.label_list = config["label_list"]
        self.max_det_results = config["max_det_results"]

    def process(self, data):
        pred = data["pred"]
        np_boxes = pred[list(pred.keys())[0]]
        if reduce(lambda x, y: x * y, np_boxes.shape) >= 6:
            keep_indexes = np_boxes[:, 1].argsort()[::-1][:
                                                          self.max_det_results]
            # TODO(gaotingquan): only support bs==1
            single_res = np_boxes[0]
            class_id = int(single_res[0])
            score = single_res[1]
            bbox = single_res[2:]
            if score > self.threshold:
                label_name = self.label_list[class_id]
                results = {
                    "class_id": class_id,
                    "score": score,
                    "bbox": bbox,
                    "label_name": label_name,
                }
                data["detection_res"] = results
                return data

        logger.warning('[Detector] No object detected.')
        results = {
            "class_id": None,
            "score": None,
            "bbox": None,
            "label_name": None,
        }
        data["detection_res"] = results
        return data
