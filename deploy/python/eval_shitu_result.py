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
import itertools
import json
import pickle

import cv2
import faiss
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from python.predict_det import DetPredictor
from python.predict_rec import RecPredictor
from utils import config, logger
from utils.draw_bbox import draw_bbox_results
from utils.get_image_list import get_image_list


def draw_pr_curve(precision,
                  recall,
                  iou=0.5,
                  out_dir='pr_curve',
                  file_name='precision_recall_curve.jpg'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_path = os.path.join(out_dir, file_name)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.error('Matplotlib not found, plaese install matplotlib.'
                     'for example: `pip install matplotlib`.')
        raise e
    plt.cla()
    plt.figure('P-R Curve')
    plt.title('Precision/Recall Curve(IoU={})'.format(iou))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.plot(recall, precision)
    plt.savefig(output_path)


class SystemPredictor(object):

    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)
        self.det_predictor = DetPredictor(config)

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
            "bbox": np.array([0, 0, shape[1],
                              shape[0]]),  # xmin, ymin, xmax, ymax
            "label_name": "foreground",
        })
        return results

    def nms_to_rec_results(self, results, thresh=0.1, cat_ids=None):
        filtered_results = []
        det_result = []
        x1 = np.array([r["bbox"][0] for r in results]).astype("float32")
        y1 = np.array([r["bbox"][1] for r in results]).astype("float32")
        w = np.array([r["bbox"][2] for r in results]).astype("float32")
        h = np.array([r["bbox"][3] for r in results]).astype("float32")
        x2 = w + x1
        y2 = h + y1
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
            if cat_ids is None:
                filtered_results.append(results[i]['rec_docs'])
            else:
                filtered_results.append(cat_ids[results[i]['rec_docs']])
            det_result.append(results[i])

        return filtered_results, det_result

    def predict(self, img, cat_dict=None):
        eval_output_dict = {}
        #  det_output_dict = {}

        # st1: get all detection results
        all_results = self.det_predictor.predict(img)
        output_list = []
        # st2: add the whole image for recognition to improve recall
        results = self.append_self(all_results, img.shape)

        # st3: recognition process, use score_thres to ensure accuracy
        for result in results:
            preds = {}
            xmin, ymin, xmax, ymax = result["bbox"].astype("int")
            crop_img = img[ymin:ymax, xmin:xmax, :].copy()
            rec_results = self.rec_predictor.predict(crop_img)
            preds["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
            scores, docs = self.Searcher.search(rec_results, self.return_k)

            # just top-1 result will be returned for the final
            if scores[0][0] >= self.config["IndexProcess"]["score_thres"]:
                preds["rec_docs"] = self.id_map[docs[0][0]].split()[1]
                preds["rec_scores"] = scores[0][0]
                preds["score"] = result["score"]
                output_list.append(preds)

        #  for i in range(100):
            #  det_thres = i / 100
            #  for j in range(1, 21, 1):
            #      length = j if len(all_results) > j else len(all_results)
            #      results_eval = copy.deepcopy(output_list[:length])
            #      for index in range(length):
            #          if results_eval[index]["score"] < det_thres:
            #              results_eval = results_eval[:index]
            #              break
            #
            #      results_eval.append(copy.deepcopy(output_list[-1]))
            #
            #      # st5: nms to the final results to avoid fetching duplicate results
            #      for k in range(100):
            #          rec_nms_thresold = k / 100
            #          eval_output, det_output = self.nms_to_rec_results(
            #              copy.deepcopy(results_eval), rec_nms_thresold,
            #              cat_dict)
            #          eval_output_dict[str(det_thres) + '_' + str(j) + "_" +
            #                           str(rec_nms_thresold)] = eval_output
            #          det_output_dict[str(det_thres) + '_' + str(j) + "_" +
        #                              str(rec_nms_thresold)] = det_output
        eval_output, det_output = self.nms_to_rec_results(output_list, 0.0, cat_dict)

        return eval_output_dict, det_output


def save_det_result(det_result_dir,
                    img_list,
                    key,
                    f1_p_r,
                    save_dir="./eval_result"):
    save_result = {}
    for x in img_list:
        with open(os.path.join(det_result_dir,
                               x.split('.')[0] + '.pkl'), 'rb') as fd:
            result = pickle.load(fd)
        result = result.get(key, None)
        save_result[x] = result

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    [det_thres, max_det, rec_nms] = [float(x) for x in key.split('_')]
    save_path = os.path.join(
        save_dir,
        "f1_{:.4f}_presion_{:.3f}_recall_{:.3f}_det_thres_{:.3f}_max_det_num_{:.3f}_rec_nms_{:.3f}"
        .format(f1_p_r[0], f1_p_r[1], f1_p_r[2], det_thres, max_det, rec_nms))
    with open(save_path, "w") as fd:
        json.dump(save_result, fd)


def cal_f1(pred,
           ground_truth,
           det_result_dir=None,
           img_list=None,
           save_dir="./eval_result"):
    max_f1 = 0
    key = None
    max_f1_p = 0
    max_f1_r = 0
    for k in pred[0].keys():
        # tp, fp, tn, fn
        pr = [0, 0, 0, 0]
        for p, g in zip(pred, ground_truth):
            gg = copy.deepcopy(g)
            for pp in p[k]:
                if pp in gg:
                    pr[0] += 1
                    gg.remove(pp)
                else:
                    pr[1] += 1
            if len(gg) != 0:
                pr[3] += 1
        p = pr[0] / (pr[0] + pr[1])
        r = pr[0] / (pr[0] + pr[3])
        if p + r == 0:
            f1 = -1
        else:
            f1 = 2 * p * r / (p + r)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_p = p
            max_f1_r = r
            key = k
        if p >= 0.9:
            [det_thres, max_det, rec_nms] = [float(x) for x in k.split('_')]
            print(
                "presion >= 0.9: presion: {:.3f}, recall: {:.3f}, f1: {:.3f}, det_thres: {:.3f}, max_det:{:.1f}, rec_nms:{:.3f}"
                .format(p, r, f1, det_thres, max_det, rec_nms))
            save_det_result(det_result_dir, img_list, k, [f1, p, r])
        if r >= 0.9:
            [det_thres, max_det, rec_nms] = [float(x) for x in k.split('_')]
            print(
                "recall >= 0.9: presion: {:.3f}, recall: {:.3f}, f1: {:.3f}, det_thres: {:.3f}, max_det:{:.1f}, rec_nms:{:.3f}"
                .format(p, r, f1, det_thres, max_det, rec_nms))
            save_det_result(det_result_dir, img_list, k, [f1, p, r])
    print("max f1: {:.4f}, presion: {:.4f}, recall: {:.4f}".format(
        max_f1, max_f1_p, max_f1_r))
    [det_thres, max_det, rec_nms] = [float(x) for x in key.split('_')]
    print("det_thres: {:.3f}, max_det:{:.1f}, rec_nms:{:.3f}".format(
        det_thres, max_det, rec_nms))
    save_det_result(det_result_dir, img_list, key,
                    [max_f1, max_f1_p, max_f1_r])


def main(config):
    system_predictor = SystemPredictor(config)
    coco = COCO(config["Global"]["infer_imgs"])
    cat_dict = {}
    for k, v in coco.cats.items():
        cat_dict[v['name']] = v['id']
    img_ids = coco.getImgIds()
    img_root_path = config['Global']['image_root_dir']

    label_map = {}
    for k, v in coco.cats.items():
        label_map[v['name']] = v['id']

    assert config["Global"]["batch_size"] == 1
    pred = []
    ground_truth = []
    img_list = []
    det_result = {}
    for img_id in img_ids:
        img_anno = coco.loadImgs(img_id)[0]
        image_file = os.path.join(img_root_path, img_anno['file_name'])
        img = cv2.imread(image_file)[:, :, ::-1]
        eval_output_dict, det_output = system_predictor.predict(
            img, cat_dict)
        #  draw_bbox_results(img, output, image_file)
        ins_anno_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        instances = coco.loadAnns(ins_anno_ids)
        gth = [x['category_id'] for x in instances]
        pred.append(eval_output_dict)
        ground_truth.append(gth)
        basename = os.path.basename(image_file)
        img_list.append(basename)
        det_result[basename] = det_output

    #  cal_f1(pred, ground_truth, det_result_save_dir, img_list)
    with open("all_results.pkl", 'wb')as fd:
        pickle.dump(det_result, fd)
    
    return


# usage:
# python python/eval_shitu.py -c configs/inference_drink.yaml \
#           -o Global.infer_imgs=/work/project/ppshit_test_data/query.json \ # coco format json file path
#           -o Global.image_root_dir=/work/dataset/ppshitu_test_data/ \  # test data root path
#           -o IndexProcess.score_thres=0
if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    config["Global"]["threshold"] = 0
    config["Global"]["max_det_results"] = 100
    config["Global"]["rec_nms_thresold"] = 0
    main(config)
