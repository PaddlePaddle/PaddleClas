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

import numpy as np

from paddle_serving_client import Client
from paddle_serving_app.reader import *
import cv2
import faiss
import os
import pickle

rec_nms_thresold = 0.05
rec_score_thres = 0.5
feature_normalize = True
return_k = 1
index_dir = "../../drink_dataset_v1.0/index"


def init_index(index_dir):
    assert os.path.exists(os.path.join(
        index_dir, "vector.index")), "vector.index not found ..."
    assert os.path.exists(os.path.join(
        index_dir, "id_map.pkl")), "id_map.pkl not found ... "

    searcher = faiss.read_index(os.path.join(index_dir, "vector.index"))

    with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
        id_map = pickle.load(fd)
    return searcher, id_map


#get box
def nms_to_rec_results(results, thresh=0.1):
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


def postprocess(fetch_dict, feature_normalize, det_boxes, searcher, id_map,
                return_k, rec_score_thres, rec_nms_thresold):
    batch_features = fetch_dict["features"]

    #do feature norm
    if feature_normalize:
        feas_norm = np.sqrt(
            np.sum(np.square(batch_features), axis=1, keepdims=True))
        batch_features = np.divide(batch_features, feas_norm)

    scores, docs = searcher.search(batch_features, return_k)

    results = []
    for i in range(scores.shape[0]):
        pred = {}
        if scores[i][0] >= rec_score_thres:
            pred["bbox"] = [int(x) for x in det_boxes[i, 2:]]
            pred["rec_docs"] = id_map[docs[i][0]].split()[1]
            pred["rec_scores"] = scores[i][0]
            results.append(pred)

    #do nms
    results = nms_to_rec_results(results, rec_nms_thresold)
    return results


#do client
if __name__ == "__main__":
    client = Client()
    client.load_client_config([
        "../../models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client",
        "../../models/general_PPLCNet_x2_5_lite_v1.0_client"
    ])
    client.connect(['127.0.0.1:9400'])

    im = cv2.imread("../../drink_dataset_v1.0/test_images/001.jpeg")
    im_shape = np.array(im.shape[:2]).reshape(-1)
    fetch_map = client.predict(
        feed={"image": im,
              "im_shape": im_shape},
        fetch=["features", "boxes"],
        batch=False)

    #add retrieval procedure
    det_boxes = fetch_map["boxes"]
    searcher, id_map = init_index(index_dir)
    results = postprocess(fetch_map, feature_normalize, det_boxes, searcher,
                          id_map, return_k, rec_score_thres, rec_nms_thresold)
    print(results)
