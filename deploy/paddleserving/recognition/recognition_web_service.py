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
from paddle_serving_server.web_service import WebService, Op
import logging
import numpy as np
import sys
import cv2
from paddle_serving_app.reader import *
import base64
import os
import faiss
import pickle
import json


class DetOp(Op):
    def init_op(self):
        self.img_preprocess = Sequential([
            BGR2RGB(), Div(255.0),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False),
            Resize((640, 640)), Transpose((2, 0, 1))
        ])

        self.img_postprocess = RCNNPostprocess("label_list.txt", "output")
        self.threshold = 0.2
        self.max_det_results = 5

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        target_size = [640, 640]
        origin_shape = im.shape[:2]
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        imgs = []
        raw_imgs = []
        for key in input_dict.keys():
            data = base64.b64decode(input_dict[key].encode('utf8'))
            raw_imgs.append(data)
            data = np.fromstring(data, np.uint8)
            raw_im = cv2.imdecode(data, cv2.IMREAD_COLOR)

            im_scale_y, im_scale_x = self.generate_scale(raw_im)
            im = self.img_preprocess(raw_im)

            im_shape = np.array(im.shape[1:]).reshape(-1)
            scale_factor = np.array([im_scale_y, im_scale_x]).reshape(-1)
            imgs.append({
                "image": im[np.newaxis, :],
                "im_shape": im_shape[np.newaxis, :],
                "scale_factor": scale_factor[np.newaxis, :],
            })
        self.raw_img = raw_imgs

        feed_dict = {
            "image": np.concatenate(
                [x["image"] for x in imgs], axis=0),
            "im_shape": np.concatenate(
                [x["im_shape"] for x in imgs], axis=0),
            "scale_factor": np.concatenate(
                [x["scale_factor"] for x in imgs], axis=0)
        }
        return feed_dict, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        boxes = self.img_postprocess(fetch_dict, visualize=False)
        boxes.sort(key=lambda x: x["score"], reverse=True)
        boxes = filter(lambda x: x["score"] >= self.threshold,
                       boxes[:self.max_det_results])
        boxes = list(boxes)
        for i in range(len(boxes)):
            boxes[i]["bbox"][2] += boxes[i]["bbox"][0] - 1
            boxes[i]["bbox"][3] += boxes[i]["bbox"][1] - 1
        result = json.dumps(boxes)
        res_dict = {"bbox_result": result, "image": self.raw_img}
        return res_dict, None, ""


class RecOp(Op):
    def init_op(self):
        self.seq = Sequential([
            BGR2RGB(), Resize((224, 224)), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                      False), Transpose((2, 0, 1))
        ])

        index_dir = "../../drink_dataset_v1.0/index"
        assert os.path.exists(os.path.join(
            index_dir, "vector.index")), "vector.index not found ..."
        assert os.path.exists(os.path.join(
            index_dir, "id_map.pkl")), "id_map.pkl not found ... "

        self.searcher = faiss.read_index(
            os.path.join(index_dir, "vector.index"))

        with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
            self.id_map = pickle.load(fd)

        self.rec_nms_thresold = 0.05
        self.rec_score_thres = 0.5
        self.feature_normalize = True
        self.return_k = 1

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        raw_img = input_dict["image"][0]
        data = np.frombuffer(raw_img, np.uint8)
        origin_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        dt_boxes = input_dict["bbox_result"]
        boxes = json.loads(dt_boxes)
        boxes.append({
            "category_id": 0,
            "score": 1.0,
            "bbox": [0, 0, origin_img.shape[1], origin_img.shape[0]]
        })
        self.det_boxes = boxes

        #construct batch images for rec
        imgs = []
        for box in boxes:
            box = [int(x) for x in box["bbox"]]
            im = origin_img[box[1]:box[3], box[0]:box[2]].copy()
            img = self.seq(im)
            imgs.append(img[np.newaxis, :].copy())

        input_imgs = np.concatenate(imgs, axis=0)
        return {"x": input_imgs}, False, None, ""

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

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        batch_features = fetch_dict["features"]

        if self.feature_normalize:
            feas_norm = np.sqrt(
                np.sum(np.square(batch_features), axis=1, keepdims=True))
            batch_features = np.divide(batch_features, feas_norm)

        scores, docs = self.searcher.search(batch_features, self.return_k)

        results = []
        for i in range(scores.shape[0]):
            pred = {}
            if scores[i][0] >= self.rec_score_thres:
                pred["bbox"] = [int(x) for x in self.det_boxes[i]["bbox"]]
                pred["rec_docs"] = self.id_map[docs[i][0]].split()[1]
                pred["rec_scores"] = scores[i][0]
                results.append(pred)

        #do nms
        results = self.nms_to_rec_results(results, self.rec_nms_thresold)
        return {"result": str(results)}, None, ""


class RecognitionService(WebService):
    def get_pipeline_response(self, read_op):
        det_op = DetOp(name="det", input_ops=[read_op])
        rec_op = RecOp(name="rec", input_ops=[det_op])
        return rec_op


product_recog_service = RecognitionService(name="recognition")
product_recog_service.prepare_pipeline_config("config.yml")
product_recog_service.run_service()
