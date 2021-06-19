# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from vector_search import Graph_Index

from utils import logger
from utils import config
from utils.get_image_list import get_image_list
from utils.draw_bbox import draw_bbox_results


class SystemPredictor(object):
    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)
        self.det_predictor = DetPredictor(config)

        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.return_k = self.config['IndexProcess']['return_k']
        self.search_budget = self.config['IndexProcess']['search_budget']

        self.Searcher = Graph_Index(
            dist_type=config['IndexProcess']['dist_type'])
        self.Searcher.load(config['IndexProcess']['index_path'])

    def append_self(self, results, shape):
        results.append({
            "class_id": 0,
            "score": 1.0,
            "bbox": np.array([0, 0, shape[1], shape[0]]),
            "label_name": "foreground",
        })
        return results

    def predict(self, img):
        output = []
        results = self.det_predictor.predict(img)
        # add the whole image for recognition
        results = self.append_self(results, img.shape)

        for result in results:
            preds = {}
            xmin, ymin, xmax, ymax = result["bbox"].astype("int")
            crop_img = img[ymin:ymax, xmin:xmax, :].copy()
            rec_results = self.rec_predictor.predict(crop_img)
            preds["bbox"] = [xmin, ymin, xmax, ymax]
            scores, docs = self.Searcher.search(
                query=rec_results,
                return_k=self.return_k,
                search_budget=self.search_budget)
            # just top-1 result will be returned for the final
            if scores[0] >= self.config["IndexProcess"]["score_thres"]:
                preds["rec_docs"] = docs[0]
                preds["rec_scores"] = scores[0]
            else:
                preds["rec_docs"] = None
                preds["rec_scores"] = 0.0

            output.append(preds)
        return output


def main(config):
    system_predictor = SystemPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    assert config["Global"]["batch_size"] == 1
    for idx, image_file in enumerate(image_list):
        img = cv2.imread(image_file)[:, :, ::-1]
        output = system_predictor.predict(img)
        draw_bbox_results(img, output, image_file)
        print(output)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
