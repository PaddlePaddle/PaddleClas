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
import faiss
import pickle
import json

from python.predict_rec import RecPredictor
from python.predict_det import DetPredictor

from utils import logger
from utils import config
from utils.get_image_list import get_image_list
from utils.draw_bbox import draw_bbox_results
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import itertools


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


def cocoapi_eval(jsonfile,
                 coco_gt,
                 style='bbox',
                 max_dets=(100, 300, 1000),
                 classwise=False,
                 sigmas=None,
                 use_area=True):
    """
    Args:
        jsonfile (str): Evaluation json file, eg: bbox.json, mask.json.
        style (str): COCOeval style, can be `bbox` , `segm` , `proposal`, `keypoints` and `keypoints_crowd`.
        coco_gt (str): Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        max_dets (tuple): COCO evaluation maxDets.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
        sigmas (nparray): keypoint labelling sigmas.
        use_area (bool): If gt annotations (eg. CrowdPose, AIC)
                         do not have 'area', please set use_area=False.
    """

    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)

    coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if classwise:
        # Compute per-category AP and PR curve
        try:
            from terminaltables import AsciiTable
        except Exception as e:
            logger.error(
                'terminaltables not found, plaese install terminaltables. '
                'for example: `pip install terminaltables`.')
            raise e
        precisions = coco_eval.eval['precision']
        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]
        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (str(nm["name"]), '{:0.3f}'.format(float(ap))))
            pr_array = precisions[0, :, idx, 0, 2]
            recall_array = np.arange(0.0, 1.01, 0.01)
            draw_pr_curve(
                pr_array,
                recall_array,
                out_dir=style + '_pr_curve',
                file_name='{}_precision_recall_curve.jpg'.format(nm["name"]))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            * [results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        logger.info('Per-category of {} AP: \n{}'.format(style, table.table))
        logger.info("per-category PR curve has output to {} folder.".format(
            style + '_pr_curve'))
    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats


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
        # st1: get all detection results
        results = self.det_predictor.predict(img)

        # st2: add the whole image for recognition to improve recall
        results = self.append_self(results, img.shape)

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
                output.append(preds)

        # st5: nms to the final results to avoid fetching duplicate results
        output = self.nms_to_rec_results(
            output, self.config["Global"]["rec_nms_thresold"])

        return output


def transform2coco(output, img_id, label_map):
    coco_results = []
    for o in output:
        coco_result = {}
        coco_result['bbox'] = [float(x) for x in o['bbox']]
        coco_result['category_id'] = int(label_map[o['rec_docs']])
        coco_result['image_id'] = int(img_id)
        coco_result['score'] = float(o['rec_scores'])
        coco_results.append(coco_result)
    return coco_results


def main(config):
    system_predictor = SystemPredictor(config)
    coco = COCO(config["Global"]["infer_imgs"])
    img_ids = coco.getImgIds()
    img_root_path = config['Global']['image_root_dir']

    label_map = {}
    for k, v in coco.cats.items():
        label_map[v['name']] = v['id']

    assert config["Global"]["batch_size"] == 1
    pred = []
    for img_id in img_ids:
        img_anno = coco.loadImgs(img_id)[0]
        image_file = os.path.join(img_root_path, img_anno['file_name'])
        img = cv2.imread(image_file)[:, :, ::-1]
        output = system_predictor.predict(img)
        #  draw_bbox_results(img, output, image_file)
        pred.extend(transform2coco(output, img_id, label_map))
    with open('result.json', 'w') as fd:
        json.dump(pred, fd)
    cocoapi_eval('result.json', coco)
    return


# usage:
# python python/eval_shitu.py -c configs/inference_drink.yaml \
#           -o Global.infer_imgs=/work/project/ppshit_test_data/query.json \ # coco format json file path
#           -o Global.image_root_dir=/work/dataset/ppshitu_test_data/ \  # test data root path
#           -o IndexProcess.score_thres=0
if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
