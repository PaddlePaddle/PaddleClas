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
from pycocotools.coco import COCO

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import copy
import cv2
import numpy as np
import faiss
import pickle

from python.predict_rec import RecPredictor

from utils import logger
from utils import config
from utils.get_image_list import get_image_list
from utils.draw_bbox import draw_bbox_results
from sklearn.metrics import f1_score


class SystemPredictor(object):
    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)

        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.return_k = self.config['IndexProcess']['return_k']

        index_dir = self.config["IndexProcess"]["index_dir"]
        assert os.path.exists(os.path.join(
            index_dir, "vector.index")), "vector.index not found ..."
        assert os.path.exists(os.path.join(
            index_dir, "id_map.pkl")), "id_map.pkl not found ... "

        if config['IndexProcess'].get("binary_index", False):
            self.Searcher = faiss.read_index_binary(
                os.path.join(index_dir, "vector.index"))
        else:
            self.Searcher = faiss.read_index(
                os.path.join(index_dir, "vector.index"))

        with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
            self.id_map = pickle.load(fd)

    def append_self(self, results, shape):
        results.append({
            "class_id": 0,
            "score": 1.0,
            "bbox":
            np.array([0, 0, shape[1], shape[0]]),  # xmin, ymin, xmax, ymax
            "label_name": "foreground",
        })
        return results

    def nms_to_rec_results(self, results, thresh=0.1):
        filtered_results = []
        x1 = np.array([r["bbox"][0] for r in results]).astype("float32")
        y1 = np.array([r["bbox"][1] for r in results]).astype("float32")
        x2 = np.array([r["bbox"][2] for r in results]).astype("float32")
        y2 = np.array([r["bbox"][3] for r in results]).astype("float32")
        scores = np.array([r["rec_scores"] for r in results])

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
            filtered_results.append(results[i])

        return filtered_results

    def predict(self, img):
        output = []
        rec_result = self.rec_predictor.predict(img)
        scores, docs = self.Searcher.search(rec_result, self.return_k)

        if scores[0][0] >= self.config["IndexProcess"]["score_thres"]:
            label = self.id_map[docs[0][0]].split()[1]
            score = scores[0][0]
            output.append([label, score])

        return output


def main(config):
    system_predictor = SystemPredictor(config)

    assert config["Global"]["batch_size"] == 1
    img_root_path = config['Global']['image_root_dir']
    predict = []
    gth = []

    coco = COCO(config["Global"]["infer_imgs"])
    img_ids = coco.getImgIds()
    label_map = {}
    for k, v in coco.cats.items():
        label_map[v['name']] = v['id']

    for img_id in img_ids:
        img_anno = coco.loadImgs(img_id)[0]
        im_fname = os.path.join(img_root_path, img_anno['file_name'])
        ins_anno_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        instances = coco.loadAnns(ins_anno_ids)
        img = cv2.imread(im_fname)
        img = img[:, :, ::-1]
        for inst in instances:
            cat_id = inst['category_id']
            bbox = [int(x) for x in inst['bbox']]
            img_obj = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            output = system_predictor.predict(img_obj)
            gth.append(cat_id)
            predict.append(label_map[output[0][0]])

    micro = f1_score(gth, predict, average='micro')
    macro = f1_score(gth, predict, average='macro')
    print("micro f1: {:.3f}".format(micro))
    print("macro f1: {:.3f}".format(macro))


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
