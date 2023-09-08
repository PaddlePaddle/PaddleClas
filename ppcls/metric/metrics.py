# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cmath import nan
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score as accuracy_metric
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import binarize

from easydict import EasyDict

from ppcls.metric.avg_metrics import AvgMetrics
from ppcls.utils.misc import AverageMeter, AttrMeter
from ppcls.utils import logger


class TopkAcc(AvgMetrics):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk
        self.reset()
        self.warned = False

    def reset(self):
        self.avg_meters = {
            f"top{k}": AverageMeter(f"top{k}")
            for k in self.topk
        }

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        output_dims = x.shape[-1]

        metric_dict = dict()
        for idx, k in enumerate(self.topk):
            if output_dims < k:
                if not self.warned:
                    msg = f"The output dims({output_dims}) is less than k({k}), so the Top-{k} metric is meaningless."
                    logger.warning(msg)
                    self.warned = True
                metric_dict[f"top{k}"] = 1
            else:
                metric_dict[f"top{k}"] = paddle.metric.accuracy(x, label, k=k)
            self.avg_meters[f"top{k}"].update(metric_dict[f"top{k}"],
                                              x.shape[0])
        return metric_dict


class mAP(nn.Layer):
    def __init__(self, descending=True):
        super().__init__()
        self.descending = descending

    def forward(self, similarities_matrix, query_img_id, gallery_img_id,
                keep_mask):
        metric_dict = dict()

        choosen_indices = paddle.argsort(
            similarities_matrix, axis=1, descending=self.descending)
        gallery_labels_transpose = paddle.transpose(gallery_img_id, [1, 0])
        gallery_labels_transpose = paddle.broadcast_to(
            gallery_labels_transpose,
            shape=[
                choosen_indices.shape[0], gallery_labels_transpose.shape[1]
            ])
        choosen_label = paddle.index_sample(gallery_labels_transpose,
                                            choosen_indices)
        equal_flag = paddle.equal(choosen_label, query_img_id)
        if keep_mask is not None:
            keep_mask = paddle.index_sample(
                keep_mask.astype('float32'), choosen_indices)
            equal_flag = paddle.logical_and(equal_flag,
                                            keep_mask.astype('bool'))
        equal_flag = paddle.cast(equal_flag, 'float32')

        num_rel = paddle.sum(equal_flag, axis=1)
        num_rel = paddle.greater_than(num_rel, paddle.to_tensor(0.))
        num_rel_index = paddle.nonzero(num_rel.astype("int"))
        num_rel_index = paddle.reshape(num_rel_index, [num_rel_index.shape[0]])

        if paddle.numel(num_rel_index).item() == 0:
            metric_dict["mAP"] = np.nan
            return metric_dict

        equal_flag = paddle.index_select(equal_flag, num_rel_index, axis=0)

        acc_sum = paddle.cumsum(equal_flag, axis=1)
        div = paddle.arange(acc_sum.shape[1]).astype("float32") + 1
        precision = paddle.divide(acc_sum, div)

        #calc map
        precision_mask = paddle.multiply(equal_flag, precision)
        ap = paddle.sum(precision_mask, axis=1) / paddle.sum(equal_flag,
                                                             axis=1)
        metric_dict["mAP"] = float(paddle.mean(ap))
        return metric_dict


class mINP(nn.Layer):
    def __init__(self, descending=True):
        super().__init__()
        self.descending = descending

    def forward(self, similarities_matrix, query_img_id, gallery_img_id,
                keep_mask):
        metric_dict = dict()

        choosen_indices = paddle.argsort(
            similarities_matrix, axis=1, descending=self.descending)
        gallery_labels_transpose = paddle.transpose(gallery_img_id, [1, 0])
        gallery_labels_transpose = paddle.broadcast_to(
            gallery_labels_transpose,
            shape=[
                choosen_indices.shape[0], gallery_labels_transpose.shape[1]
            ])
        choosen_label = paddle.index_sample(gallery_labels_transpose,
                                            choosen_indices)
        equal_flag = paddle.equal(choosen_label, query_img_id)
        if keep_mask is not None:
            keep_mask = paddle.indechmx_sample(
                keep_mask.astype('float32'), choosen_indices)
            equal_flag = paddle.logical_and(equal_flag,
                                            keep_mask.astype('bool'))
        equal_flag = paddle.cast(equal_flag, 'float32')

        num_rel = paddle.sum(equal_flag, axis=1)
        num_rel = paddle.greater_than(num_rel, paddle.to_tensor(0.))
        num_rel_index = paddle.nonzero(num_rel.astype("int"))
        num_rel_index = paddle.reshape(num_rel_index, [num_rel_index.shape[0]])
        equal_flag = paddle.index_select(equal_flag, num_rel_index, axis=0)

        #do accumulative sum
        div = paddle.arange(equal_flag.shape[1]).astype("float32") + 2
        minus = paddle.divide(equal_flag, div)
        auxilary = paddle.subtract(equal_flag, minus)
        hard_index = paddle.argmax(auxilary, axis=1).astype("float32")
        all_INP = paddle.divide(paddle.sum(equal_flag, axis=1), hard_index)
        mINP = paddle.mean(all_INP)
        metric_dict["mINP"] = float(mINP)
        return metric_dict


class TprAtFpr(nn.Layer):
    def __init__(self, max_fpr=1 / 1000.):
        super().__init__()
        self.gt_pos_score_list = []
        self.gt_neg_score_list = []
        self.softmax = nn.Softmax(axis=-1)
        self.max_fpr = max_fpr
        self.max_tpr = 0.

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        x = self.softmax(x)
        for i, label_i in enumerate(label):
            if label_i[0] == 0:
                self.gt_neg_score_list.append(x[i][1].numpy())
            else:
                self.gt_pos_score_list.append(x[i][1].numpy())
        return {}

    def reset(self):
        self.gt_pos_score_list = []
        self.gt_neg_score_list = []
        self.max_tpr = 0.

    @property
    def avg(self):
        return self.max_tpr

    @property
    def avg_info(self):
        max_tpr = 0.
        result = ""
        gt_pos_score_list = np.array(self.gt_pos_score_list)
        gt_neg_score_list = np.array(self.gt_neg_score_list)
        for i in range(0, 10000):
            threshold = i / 10000.
            if len(gt_pos_score_list) == 0:
                continue
            tpr = np.sum(
                gt_pos_score_list > threshold) / len(gt_pos_score_list)
            if len(gt_neg_score_list) == 0 and tpr > max_tpr:
                max_tpr = tpr
                result = "threshold: {}, fpr: 0.0, tpr: {:.5f}".format(
                    threshold, tpr)
                msg = f"The number of negative samples is 0, please add negative samples."
                logger.warning(msg)
            fpr = np.sum(
                gt_neg_score_list > threshold) / len(gt_neg_score_list)
            if fpr <= self.max_fpr and tpr > max_tpr:
                max_tpr = tpr
                result = "threshold: {}, fpr: {}, tpr: {:.5f}".format(
                    threshold, fpr, tpr)
        self.max_tpr = max_tpr
        return result


class MultilabelMeanAccuracy(nn.Layer):
    def __init__(self,
                 start_threshold=0.4,
                 num_iterations=10,
                 end_threshold=0.9):
        super().__init__()
        self.start_threshold = start_threshold
        self.num_iterations = num_iterations
        self.end_threshold = end_threshold
        self.gt_all_score_list = []
        self.gt_label_score_list = []
        self.max_acc = 0.

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        x = F.sigmoid(x)
        label = label[:, 0, :]
        for i in range(len(x)):
            self.gt_all_score_list.append(x[i].numpy())
            self.gt_label_score_list.append(label[i].numpy())
        return {}

    def reset(self):
        self.gt_all_score_list = []
        self.gt_label_score_list = []
        self.max_acc = 0.

    @property
    def avg(self):
        return self.max_acc

    @property
    def avg_info(self):
        max_acc = 0.
        result = ""
        gt_all_score_list = np.array(self.gt_all_score_list)
        gt_label_score_list = np.array(self.gt_label_score_list)
        for i in range(self.num_iterations):
            threshold = self.start_threshold + i * (self.end_threshold -
                                                    self.start_threshold
                                                    ) / self.num_iterations
            pred_label = (gt_all_score_list > threshold).astype(int)
            TP = np.sum(
                (gt_label_score_list == 1) * (pred_label == 1)).astype(float)
            TN = np.sum(
                (gt_label_score_list == 0) * (pred_label == 0)).astype(float)
            acc = (TP + TN) / len(gt_all_score_list)
            if max_acc <= acc:
                max_acc = acc
                result = "threshold: {}, mean_acc: {}".format(
                    threshold, max_acc / len(gt_label_score_list[0]))
        self.max_acc = max_acc / len(gt_label_score_list[0])
        return result


class Recallk(nn.Layer):
    def __init__(self, topk=(1, 5), descending=True):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk
        self.descending = descending

    def forward(self, similarities_matrix, query_img_id, gallery_img_id,
                keep_mask):
        metric_dict = dict()

        # get cmc
        choosen_indices = paddle.argsort(
            similarities_matrix, axis=1, descending=self.descending)
        gallery_labels_transpose = gallery_img_id.t()
        gallery_labels_transpose = paddle.broadcast_to(
            gallery_labels_transpose,
            shape=[
                choosen_indices.shape[0], gallery_labels_transpose.shape[1]
            ])
        choosen_label = paddle.index_sample(gallery_labels_transpose,
                                            choosen_indices)
        equal_flag = paddle.equal(choosen_label, query_img_id)
        if keep_mask is not None:
            keep_mask = paddle.index_sample(
                keep_mask.astype("float32"), choosen_indices)
            equal_flag = equal_flag & keep_mask.astype("bool")
        equal_flag = paddle.cast(equal_flag, "float32")
        real_query_num = paddle.sum(equal_flag, axis=1)
        real_query_num = paddle.sum((real_query_num > 0.0).astype("float32"))

        acc_sum = paddle.cumsum(equal_flag, axis=1)
        mask = (acc_sum > 0.0).astype("float32")
        all_cmc = (paddle.sum(mask, axis=0) / real_query_num).numpy()

        for k in self.topk:
            metric_dict["recall{}".format(k)] = all_cmc[k - 1]
        return metric_dict


class Precisionk(nn.Layer):
    def __init__(self, topk=(1, 5), descending=True):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk
        self.descending = descending

    def forward(self, similarities_matrix, query_img_id, gallery_img_id,
                keep_mask):
        metric_dict = dict()

        #get cmc
        choosen_indices = paddle.argsort(
            similarities_matrix, axis=1, descending=self.descending)
        gallery_labels_transpose = paddle.transpose(gallery_img_id, [1, 0])
        gallery_labels_transpose = paddle.broadcast_to(
            gallery_labels_transpose,
            shape=[
                choosen_indices.shape[0], gallery_labels_transpose.shape[1]
            ])
        choosen_label = paddle.index_sample(gallery_labels_transpose,
                                            choosen_indices)
        equal_flag = paddle.equal(choosen_label, query_img_id)
        if keep_mask is not None:
            keep_mask = paddle.index_sample(
                keep_mask.astype('float32'), choosen_indices)
            equal_flag = paddle.logical_and(equal_flag,
                                            keep_mask.astype('bool'))
        equal_flag = paddle.cast(equal_flag, 'float32')

        Ns = paddle.arange(gallery_img_id.shape[0]) + 1
        equal_flag_cumsum = paddle.cumsum(equal_flag, axis=1)
        Precision_at_k = (paddle.mean(equal_flag_cumsum, axis=0) / Ns).numpy()

        for k in self.topk:
            metric_dict["precision@{}".format(k)] = Precision_at_k[k - 1]

        return metric_dict


class DistillationTopkAcc(TopkAcc):
    def __init__(self, model_key, feature_key=None, topk=(1, 5)):
        super().__init__(topk=topk)
        self.model_key = model_key
        self.feature_key = feature_key

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x[self.model_key]
        if self.feature_key is not None:
            x = x[self.feature_key]
        return super().forward(x, label)


class GoogLeNetTopkAcc(TopkAcc):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        return super().forward(x[0], label)


class MultiLabelMetric(AvgMetrics):
    def __init__(self, bi_threshold=0.5):
        super().__init__()
        self.bi_threshold = bi_threshold

    def _multi_hot_encode(self, output):
        logits = F.sigmoid(output).numpy()
        return binarize(logits, threshold=self.bi_threshold)


class HammingDistance(MultiLabelMetric):
    """
    Soft metric based label for multilabel classification
    Returns:
        The smaller the return value is, the better model is.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.avg_meters = {"HammingDistance": AverageMeter("HammingDistance")}

    def forward(self, output, target):
        preds = super()._multi_hot_encode(output)
        metric_dict = dict()
        metric_dict["HammingDistance"] = paddle.to_tensor(
            hamming_loss(target, preds))
        self.avg_meters["HammingDistance"].update(
            float(metric_dict["HammingDistance"]), output.shape[0])
        return metric_dict


class AccuracyScore(MultiLabelMetric):
    """
    Hard metric for multilabel classification
    Args:
        base: ["sample", "label"], default="sample"
            if "sample", return metric score based sample,
            if "label", return metric score based label.
    Returns:
        accuracy:
    """

    def __init__(self, base="label"):
        super().__init__()
        assert base in ["sample", "label"
                        ], 'must be one of ["sample", "label"]'
        self.base = base
        self.reset()

    def reset(self):
        self.avg_meters = {"AccuracyScore": AverageMeter("AccuracyScore")}

    def forward(self, output, target):
        preds = super()._multi_hot_encode(output)
        metric_dict = dict()
        if self.base == "sample":
            accuracy = accuracy_metric(target, preds)
        elif self.base == "label":
            mcm = multilabel_confusion_matrix(target, preds)
            tns = mcm[:, 0, 0]
            fns = mcm[:, 1, 0]
            tps = mcm[:, 1, 1]
            fps = mcm[:, 0, 1]
            accuracy = (sum(tps) + sum(tns)) / (
                sum(tps) + sum(tns) + sum(fns) + sum(fps))
        metric_dict["AccuracyScore"] = paddle.to_tensor(accuracy)
        self.avg_meters["AccuracyScore"].update(
            float(metric_dict["AccuracyScore"]), output.shape[0])
        return metric_dict


def get_attr_metrics(gt_label, preds_probs, threshold):
    """
    index: evaluated label index
    adapted from "https://github.com/valencebond/Rethinking_of_PAR/blob/master/metrics/pedestrian_metrics.py"
    """
    pred_label = (preds_probs > threshold).astype(int)

    eps = 1e-20
    result = EasyDict()

    has_fuyi = gt_label == -1
    pred_label[has_fuyi] = -1

    ###############################
    # label metrics
    # TP + FN
    result.gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    result.gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    result.true_pos = np.sum((gt_label == 1) * (pred_label == 1),
                             axis=0).astype(float)
    # TN
    result.true_neg = np.sum((gt_label == 0) * (pred_label == 0),
                             axis=0).astype(float)
    # FP
    result.false_pos = np.sum(((gt_label == 0) * (pred_label == 1)),
                              axis=0).astype(float)
    # FN
    result.false_neg = np.sum(((gt_label == 1) * (pred_label == 0)),
                              axis=0).astype(float)

    ################
    # instance metrics
    result.gt_pos_ins = np.sum((gt_label == 1), axis=1).astype(float)
    result.true_pos_ins = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    result.intersect_pos = np.sum((gt_label == 1) * (pred_label == 1),
                                  axis=1).astype(float)
    # IOU
    result.union_pos = np.sum(((gt_label == 1) + (pred_label == 1)),
                              axis=1).astype(float)

    return result


class ATTRMetric(nn.Layer):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def reset(self):
        self.attrmeter = AttrMeter(threshold=0.5)

    def forward(self, output, target):
        metric_dict = get_attr_metrics(target[:, 0, :].numpy(),
                                       output.numpy(), self.threshold)
        self.attrmeter.update(metric_dict)
        return metric_dict


class MultiLabelMAP(nn.Layer):
    """
    Calculate multi-label classification mean average precision.
    Currently, support two types: 11point and integral

    The code base on:
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/metrics/map_utils.py

    Args:
        map_type (str): Calculation method of mean average.
    """

    def __init__(self, map_type='integral'):
        super().__init__()
        assert map_type in ['11point', 'integral'], \
            "map_type currently only support '11point' and 'integral'"
        self.map_type = map_type

        self.reset()

    def reset(self):
        self.is_latest = True
        self.class_score_poss = None
        self.class_gt_counts = None
        self.mAP = 0.0

    def one_class_update(self, score, gt_label, class_idx):
        topk_idx = np.argsort(score)[::-1]
        topk_score = score[topk_idx]
        topk_gt_label = gt_label[topk_idx]
        for s, l in zip(topk_score, topk_gt_label):
            if int(l) == 1:
                self.class_score_poss[class_idx].append([s, 1.])
                self.class_gt_counts[class_idx] += 1
            else:
                self.class_score_poss[class_idx].append([s, 0.])

    @staticmethod
    def get_tp_fp_accum(score_pos_list):
        """
        Calculate accumulating true/false positive results from
        [score, pos] records
        """
        sorted_list = sorted(score_pos_list, key=lambda s: s[0], reverse=True)

        accum_tp = 0
        accum_fp = 0
        accum_tp_list = []
        accum_fp_list = []
        for (score, pos) in sorted_list:
            accum_tp += int(pos)
            accum_tp_list.append(accum_tp)
            accum_fp += 1 - int(pos)
            accum_fp_list.append(accum_fp)

        return accum_tp_list, accum_fp_list

    def compute_mAP(self):
        if not self.is_latest:
            mAP = 0.
            valid_cnt = 0
            for score_pos, count in zip(self.class_score_poss,
                                        self.class_gt_counts):
                if count == 0:
                    continue

                if len(score_pos) == 0:
                    valid_cnt += 1
                    continue

                accum_tp_list, accum_fp_list = \
                    self.get_tp_fp_accum(score_pos)
                precision = []
                recall = []
                for ac_tp, ac_fp in zip(accum_tp_list, accum_fp_list):
                    precision.append(float(ac_tp) / (ac_tp + ac_fp))
                    recall.append(float(ac_tp) / count)

                one_class_ap = 0.0
                if self.map_type == '11point':
                    max_precisions = [0.] * 11
                    start_idx = len(precision) - 1
                    for j in range(10, -1, -1):
                        for i in range(start_idx, -1, -1):
                            if recall[i] < float(j) / 10.:
                                start_idx = i
                                if j > 0:
                                    max_precisions[j - 1] = max_precisions[j]
                                    break
                            else:
                                if max_precisions[j] < precision[i]:
                                    max_precisions[j] = precision[i]
                    one_class_ap = sum(max_precisions) / 11.
                    mAP += one_class_ap
                    valid_cnt += 1
                elif self.map_type == 'integral':
                    import math
                    prev_recall = 0.
                    for i in range(len(precision)):
                        recall_gap = math.fabs(recall[i] - prev_recall)
                        if recall_gap > 1e-6:
                            one_class_ap += precision[i] * recall_gap
                            prev_recall = recall[i]
                    mAP += one_class_ap
                    valid_cnt += 1
                else:
                    raise NotImplementedError(
                        f"Unsupported mAP type {self.map_type}")

            self.mAP = mAP / float(valid_cnt) if valid_cnt > 0 else mAP

            self.is_latest = True

    def forward(self, output, target):
        scores = F.sigmoid(output).numpy()
        gt_labels = target.numpy()

        if self.class_score_poss is None:
            self.class_score_poss = [[] for _ in range(scores.shape[-1])]
        if self.class_gt_counts is None:
            self.class_gt_counts = [0] * scores.shape[-1]

        for class_idx in range(scores.shape[-1]):
            score = scores[:, class_idx]
            gt_label = gt_labels[:, class_idx]
            self.one_class_update(score, gt_label, class_idx)

        self.is_latest = False

        return {}

    @property
    def avg_info(self):
        self.compute_mAP()
        return f"MultiLabelMAP({self.map_type}): {self.mAP:.3f}"

    @property
    def avg(self):
        self.compute_mAP()
        return self.mAP
