import os

import cv2
import numpy as np
import faiss
import pickle

from paddleclas.deploy.utils import logger, config
from paddleclas.deploy.utils.get_image_list import get_image_and_label_list
from paddleclas.deploy.python.build_gallery import GalleryBuilder
from paddleclas.deploy.python.predict_rec import RecPredictor
from paddleclas.deploy.python.predict_det import DetPredictor


class SystemPredictor(object):
    def __init__(self, config):
        self.config = config
        self.det_predictor = DetPredictor(config)
        self.rec_predictor = RecPredictor(config)

        # create searcher
        self.return_k = self.config['IndexProcess']['return_k']
        self.index_dir = self.config['IndexProcess']['index_dir']
        if config['IndexProcess'].get("binary_index", False):
            self.Searcher = faiss.read_index_binary(
                os.path.join(self.index_dir, "vector.index"))
        else:
            self.Searcher = faiss.read_index(
                os.path.join(self.index_dir, "vector.index"))

        with open(os.path.join(self.index_dir, "id_map.pkl"), "rb") as fd:
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

    def sort_output_by_scores(self, outputs_list, scores_list):
        scores_list = np.array(scores_list)
        order = scores_list.argsort()[::-1]
        outputs = []
        for idx in order:
            outputs.append(outputs_list[idx])
        return outputs

    def predict(self, img):
        all_det_results = self.det_predictor.predict(img)
        results = self.append_self(all_det_results, img.shape)

        outputs_list = []
        scores_list = []
        for result in results:
            preds = {}
            xmin, ymin, xmax, ymax = result["bbox"].astype("int")
            crop_img = img[ymin:ymax, xmin:xmax, :].copy()
            rec_results = self.rec_predictor.predict(crop_img)
            scores, docs = self.Searcher.search(rec_results, self.return_k)

            outputs_list.append(self.id_map[docs[0][0]].split()[1])
            scores_list.append(scores[0][0])
        outputs = self.sort_output_by_scores(outputs_list, scores_list)

        return outputs


def get_recall(gth, pred):
    assert len(gth) == len(pred)
    recall_list = [0] * len(pred[0])
    for g, p in zip(gth, pred):
        for i in range(len(pred[0])):
            if g in p[:i + 1]:
                recall_list[i] += 1
    recall_list = [x / len(pred) for x in recall_list]
    return recall_list


def main(config):
    # build gallery
    assert "IndexProcess" in config.keys(), "Index config not found ... "
    operation_method = config["IndexProcess"].get("index_operation",
                                                  "new").lower()
    assert operation_method == "new", "The operation should be 'new' during evaluating."

    GalleryBuilder(config)

    syster_predictor = SystemPredictor(config)

    # get images
    assert "Eval" in config.keys(), "Eval config not found ... "
    eval_imgs_list, eval_gth = get_image_and_label_list(
        config["Eval"]["image_root"], config["Eval"]["cls_label_path"])

    # create output file
    assert "output_dir" in config['Eval'].keys(
    ), "Output dir config not found ... "
    output_dir = config['Eval']["output_dir"]
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    results_file = open(os.path.join(output_dir, 'eval_resutls.txt'), 'a+')
    results_file.write("Dataset name: %s\n" % (config['Eval']['name']))

    # evaluation
    predict = []
    for img_name in eval_imgs_list:
        img = cv2.imread(img_name)
        img = img[:, :, ::-1]
        output = syster_predictor.predict(img)

        predict.append(output)

    recall_list = get_recall(eval_gth, predict)
    for i, x in enumerate(recall_list):
        print("recal_{}: {:0.4f}".format(i + 1, x))
        results_file.write("recal_{}: {:0.4f}\n".format(i + 1, x))
    results_file.write('\n')


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
