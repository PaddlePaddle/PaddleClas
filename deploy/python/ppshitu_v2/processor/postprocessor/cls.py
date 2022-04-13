import os

import numpy as np

from utils import logger
from ..base_processor import BaseProcessor


class TopK(BaseProcessor):
    def __init__(self, config):
        super(PaddlePredictor, self).__init__(config)
        self.topk = config["k"]
        assert isinstance(self.topk, (int, ))

        class_id_map_file = config["class_id_map_file"]
        self.class_id_map = self.parse_class_id_map(class_id_map_file)

        self.multilabel = config.get("multilabel", False)

    def parse_class_id_map(self, class_id_map_file):
        if class_id_map_file is None:
            return None

        if not os.path.exists(class_id_map_file):
            logger.warning(
                "[Classification] If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!"
            )
            return None

        try:
            class_id_map = {}
            with open(class_id_map_file, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    partition = line.split("\n")[0].partition(" ")
                    class_id_map[int(partition[0])] = str(partition[-1])
        except Exception as ex:
            logger.warning(f"[Classification] {ex}")
            class_id_map = None
        return class_id_map

    def process(self, data):
        logits = data["pred"]["logits"]
        all_results = []
        for probs in logits:
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype(
                "int32") if not self.multilabel else np.where(
                    probs >= 0.5)[0].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])
            result = {
                "class_ids": clas_id_list,
                "scores": np.around(
                    score_list, decimals=5).tolist(),
            }
            if label_name_list is not None:
                result["label_names"] = label_name_list
            all_results.append(result)

        data["classification_res"] = all_results
        return data
