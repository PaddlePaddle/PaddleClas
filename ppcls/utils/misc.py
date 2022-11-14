# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle

__all__ = ['AverageMeter']


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Code was based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name='', fmt='f', postfix="", need_avg=True):
        self.name = name
        self.fmt = fmt
        self.postfix = postfix
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def avg_info(self):
        if isinstance(self.avg, paddle.Tensor):
            self.avg = float(self.avg)
        return "{}: {:.5f}".format(self.name, self.avg)

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}{self.postfix}'.format(
            self=self)

    @property
    def total_minute(self):
        return '{self.name} {s:{self.fmt}}{self.postfix} min'.format(
            s=self.sum / 60, self=self)

    @property
    def mean(self):
        return '{self.name}: {self.avg:{self.fmt}}{self.postfix}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        return '{self.name}: {self.val:{self.fmt}}{self.postfix}'.format(
            self=self)


class AttrMeter(object):
    """
    Computes and stores the average and current value
    Code was based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.gt_pos = 0
        self.gt_neg = 0
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0

        self.gt_pos_ins = []
        self.true_pos_ins = []
        self.intersect_pos = []
        self.union_pos = []

    def update(self, metric_dict):
        self.gt_pos += metric_dict['gt_pos']
        self.gt_neg += metric_dict['gt_neg']
        self.true_pos += metric_dict['true_pos']
        self.true_neg += metric_dict['true_neg']
        self.false_pos += metric_dict['false_pos']
        self.false_neg += metric_dict['false_neg']

        self.gt_pos_ins += metric_dict['gt_pos_ins'].tolist()
        self.true_pos_ins += metric_dict['true_pos_ins'].tolist()
        self.intersect_pos += metric_dict['intersect_pos'].tolist()
        self.union_pos += metric_dict['union_pos'].tolist()

    def res(self):
        import numpy as np
        eps = 1e-20
        label_pos_recall = 1.0 * self.true_pos / (
            self.gt_pos + eps)  # true positive
        label_neg_recall = 1.0 * self.true_neg / (
            self.gt_neg + eps)  # true negative
        # mean accuracy
        label_ma = (label_pos_recall + label_neg_recall) / 2

        label_pos_recall = np.mean(label_pos_recall)
        label_neg_recall = np.mean(label_neg_recall)
        label_prec = (self.true_pos / (self.true_pos + self.false_pos + eps))
        label_acc = (self.true_pos /
                     (self.true_pos + self.false_pos + self.false_neg + eps))
        label_f1 = np.mean(2 * label_prec * label_pos_recall /
                           (label_prec + label_pos_recall + eps))

        ma = (np.mean(label_ma))

        self.gt_pos_ins = np.array(self.gt_pos_ins)
        self.true_pos_ins = np.array(self.true_pos_ins)
        self.intersect_pos = np.array(self.intersect_pos)
        self.union_pos = np.array(self.union_pos)
        instance_acc = self.intersect_pos / (self.union_pos + eps)
        instance_prec = self.intersect_pos / (self.true_pos_ins + eps)
        instance_recall = self.intersect_pos / (self.gt_pos_ins + eps)
        instance_f1 = 2 * instance_prec * instance_recall / (
            instance_prec + instance_recall + eps)

        instance_acc = np.mean(instance_acc)
        instance_prec = np.mean(instance_prec)
        instance_recall = np.mean(instance_recall)
        instance_f1 = 2 * instance_prec * instance_recall / (
            instance_prec + instance_recall + eps)

        instance_acc = np.mean(instance_acc)
        instance_prec = np.mean(instance_prec)
        instance_recall = np.mean(instance_recall)
        instance_f1 = np.mean(instance_f1)

        res = [
            ma, label_f1, label_pos_recall, label_neg_recall, instance_f1,
            instance_acc, instance_prec, instance_recall
        ]
        return res
