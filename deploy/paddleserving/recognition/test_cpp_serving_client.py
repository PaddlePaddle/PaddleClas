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

import sys
import numpy as np

from paddle_serving_client import Client
from paddle_serving_app.reader import *
import cv2
import faiss
import os
import pickle


class MainbodyDetect():
    """
    pp-shitu mainbody detect.
    include preprocess, process, postprocess
    return detect results
    Attention: Postprocess include num limit and box filter; no nms 
    """

    def __init__(self):
        self.preprocess = DetectionSequential([
            DetectionFile2Image(), DetectionNormalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            DetectionResize(
                (640, 640), False, interpolation=2), DetectionTranspose(
                    (2, 0, 1))
        ])

        self.client = Client()
        self.client.load_client_config(
            "../../models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/serving_client_conf.prototxt"
        )
        self.client.connect(['127.0.0.1:9293'])

        self.max_det_result = 5
        self.conf_threshold = 0.2

    def predict(self, imgpath):
        im, im_info = self.preprocess(imgpath)
        im_shape = np.array(im.shape[1:]).reshape(-1)
        scale_factor = np.array(list(im_info['scale_factor'])).reshape(-1)

        fetch_map = self.client.predict(
            feed={
                "image": im,
                "im_shape": im_shape,
                "scale_factor": scale_factor,
            },
            fetch=["save_infer_model/scale_0.tmp_1"],
            batch=False)
        return self.postprocess(fetch_map, imgpath)

    def postprocess(self, fetch_map, imgpath):
        #1. get top max_det_result
        det_results = fetch_map["save_infer_model/scale_0.tmp_1"]
        if len(det_results) > self.max_det_result:
            boxes_reserved = fetch_map[
                "save_infer_model/scale_0.tmp_1"][:self.max_det_result]
        else:
            boxes_reserved = det_results

        #2. do conf threshold
        boxes_list = []
        for i in range(boxes_reserved.shape[0]):
            if (boxes_reserved[i, 1]) > self.conf_threshold:
                boxes_list.append(boxes_reserved[i, :])

        #3. add origin image box
        origin_img = cv2.imread(imgpath)
        boxes_list.append(
            np.array([0, 1.0, 0, 0, origin_img.shape[1], origin_img.shape[0]]))
        return np.array(boxes_list)


class ObjectRecognition():
    """
    pp-shitu object recognion for all objects detected by MainbodyDetect.
    include preprocess, process, postprocess
    preprocess include preprocess for each image and batching.
    Batch process
    postprocess include retrieval and nms
    """

    def __init__(self):
        self.client = Client()
        self.client.load_client_config(
            "../../models/general_PPLCNet_x2_5_lite_v1.0_client/serving_client_conf.prototxt"
        )
        self.client.connect(["127.0.0.1:9294"])

        self.seq = Sequential([
            BGR2RGB(), Resize((224, 224)), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                      False), Transpose((2, 0, 1))
        ])

        self.searcher, self.id_map = self.init_index()

        self.rec_nms_thresold = 0.05
        self.rec_score_thres = 0.5
        self.feature_normalize = True
        self.return_k = 1

    def init_index(self):
        index_dir = "../../drink_dataset_v1.0/index"
        assert os.path.exists(os.path.join(
            index_dir, "vector.index")), "vector.index not found ..."
        assert os.path.exists(os.path.join(
            index_dir, "id_map.pkl")), "id_map.pkl not found ... "

        searcher = faiss.read_index(os.path.join(index_dir, "vector.index"))

        with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
            id_map = pickle.load(fd)
        return searcher, id_map

    def predict(self, det_boxes, imgpath):
        #1. preprocess
        batch_imgs = []
        origin_img = cv2.imread(imgpath)
        for i in range(det_boxes.shape[0]):
            box = det_boxes[i]
            x1, y1, x2, y2 = [int(x) for x in box[2:]]
            cropped_img = origin_img[y1:y2, x1:x2, :].copy()
            tmp = self.seq(cropped_img)
            batch_imgs.append(tmp)
        batch_imgs = np.array(batch_imgs)

        #2. process
        fetch_map = self.client.predict(
            feed={"x": batch_imgs}, fetch=["features"], batch=True)
        batch_features = fetch_map["features"]

        #3. postprocess
        if self.feature_normalize:
            feas_norm = np.sqrt(
                np.sum(np.square(batch_features), axis=1, keepdims=True))
            batch_features = np.divide(batch_features, feas_norm)
        scores, docs = self.searcher.search(batch_features, self.return_k)

        results = []
        for i in range(scores.shape[0]):
            pred = {}
            if scores[i][0] >= self.rec_score_thres:
                pred["bbox"] = [int(x) for x in det_boxes[i, 2:]]
                pred["rec_docs"] = self.id_map[docs[i][0]].split()[1]
                pred["rec_scores"] = scores[i][0]
                results.append(pred)
        return self.nms_to_rec_results(results)

    def nms_to_rec_results(self, results):
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
            inds = np.where(ovr <= self.rec_nms_thresold)[0]
            order = order[inds + 1]
            filtered_results.append(results[i])
        return filtered_results


if __name__ == "__main__":
    det = MainbodyDetect()
    rec = ObjectRecognition()

    #1. get det_results    
    imgpath = "../../drink_dataset_v1.0/test_images/001.jpeg"
    det_results = det.predict(imgpath)

    #2. get rec_results
    rec_results = rec.predict(det_results, imgpath)
    print(rec_results)
