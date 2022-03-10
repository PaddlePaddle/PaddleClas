from functools import reduce

import numpy as np


class DetPostProcessor(object):
    def __init__(self, config):
        super().__init__()
        self.threshold = config["threshold"]
        self.label_list = config["label_list"]
        self.max_det_results = config["max_det_results"]

    def process(self, pred):
        np_boxes = pred["save_infer_model/scale_0.tmp_1"]
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            np_boxes = np.array([])

        keep_indexes = np_boxes[:, 1].argsort()[::-1][:self.max_det_results]
        results = []
        for idx in keep_indexes:
            single_res = np_boxes[idx]
            class_id = int(single_res[0])
            score = single_res[1]
            bbox = single_res[2:]
            if score < self.threshold:
                continue
            label_name = self.label_list[class_id]
            results.append({
                "class_id": class_id,
                "score": score,
                "bbox": bbox,
                "label_name": label_name,
            })
        return results
