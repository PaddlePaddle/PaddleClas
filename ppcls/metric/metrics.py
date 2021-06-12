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

import numpy as np
import paddle
import paddle.nn as nn
from functools import lru_cache


class TopkAcc(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        metric_dict = dict()
        for k in self.topk:
            metric_dict["top{}".format(k)] = paddle.metric.accuracy(
                x, label, k=k)
        return metric_dict


class mAP(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, similarities_matrix, query_img_id, gallery_img_id):
        metric_dict = dict()
        _, all_AP, _ = get_metrics(similarities_matrix, query_img_id,
                                   gallery_img_id)

        mAP = np.mean(all_AP)
        metric_dict["mAP"] = mAP
        return metric_dict


class mINP(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, similarities_matrix, query_img_id, gallery_img_id):
        metric_dict = dict()
        _, _, all_INP = get_metrics(similarities_matrix, query_img_id,
                                    gallery_img_id)

        mINP = np.mean(all_INP)
        metric_dict["mINP"] = mINP
        return metric_dict


class Recallk(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk
        self.max_rank = max(self.topk) if max(self.topk) > 50 else 50

    def forward(self, similarities_matrix, query_img_id, gallery_img_id):
        metric_dict = dict()
        all_cmc, _, _ = get_metrics(similarities_matrix, query_img_id,
                                    gallery_img_id, self.max_rank)

        for k in self.topk:
            metric_dict["recall{}".format(k)] = all_cmc[k - 1]
        return metric_dict


# retrieval metrics
class RetriMetric(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_rank = 50  #max(self.topk) if max(self.topk) > 50 else 50

    def forward(self, similarities_matrix, query_img_id, gallery_img_id):
        metric_dict = dict()
        all_cmc, all_AP, all_INP = get_metrics(
            similarities_matrix, query_img_id, gallery_img_id, self.max_rank)
        if "Recallk" in self.config.keys():
            topk = self.config['Recallk']['topk']
            assert isinstance(topk, (int, list, tuple))
            if isinstance(topk, int):
                topk = [topk]
            for k in topk:
                metric_dict["recall{}".format(k)] = all_cmc[k - 1]
        if "mAP" in self.config.keys():
            mAP = np.mean(all_AP)
            metric_dict["mAP"] = mAP
        if "mINP" in self.config.keys():
            mINP = np.mean(all_INP)
            metric_dict["mINP"] = mINP
        return metric_dict


@lru_cache()
def get_metrics(similarities_matrix, query_img_id, gallery_img_id,
                max_rank=50):
    num_q, num_g = similarities_matrix.shape
    q_pids = query_img_id.numpy().reshape((query_img_id.shape[0]))
    g_pids = gallery_img_id.numpy().reshape((gallery_img_id.shape[0]))
    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(
            num_g))
    indices = paddle.argsort(
        similarities_matrix, axis=1, descending=True).numpy()

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    for q_idx in range(num_q):
        raw_cmc = matches[q_idx]
        if not np.any(raw_cmc):
            continue
        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP


class DistillationTopkAcc(TopkAcc):
    def __init__(self, model_key, feature_key=None, topk=(1, 5)):
        super().__init__(topk=topk)
        self.model_key = model_key
        self.feature_key = feature_key

    def forward(self, x, label):
        x = x[self.model_key]
        if self.feature_key is not None:
            x = x[self.feature_key]
        return super().forward(x, label)
