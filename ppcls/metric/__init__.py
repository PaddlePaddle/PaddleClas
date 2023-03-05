#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import copy
from collections import OrderedDict

from .avg_metrics import AvgMetrics
from .metrics import TopkAcc, mAP, mINP, Recallk, Precisionk
from .metrics import DistillationTopkAcc
from .metrics import GoogLeNetTopkAcc
from .metrics import HammingDistance, AccuracyScore
from .metrics import ATTRMetric
from .metrics import TprAtFpr, MultilabelMeanAccuracy


class CombinedMetrics(AvgMetrics):
    def __init__(self, config_list):
        super().__init__()
        self.metric_func_list = []
        assert isinstance(config_list, list), (
            'operator config should be a list')
        for config in config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            metric_name = list(config)[0]
            metric_params = config[metric_name]
            if metric_params is not None:
                self.metric_func_list.append(
                    eval(metric_name)(**metric_params))
            else:
                self.metric_func_list.append(eval(metric_name)())
        self.reset()

    def forward(self, *args, **kwargs):
        metric_dict = OrderedDict()
        for idx, metric_func in enumerate(self.metric_func_list):
            metric_dict.update(metric_func(*args, **kwargs))
        return metric_dict

    @property
    def avg_info(self):
        return ", ".join([metric.avg_info for metric in self.metric_func_list])

    @property
    def avg(self):
        return self.metric_func_list[0].avg

    def attr_res(self):
        return self.metric_func_list[0].attrmeter.res()

    def reset(self):
        for metric in self.metric_func_list:
            if hasattr(metric, "reset"):
                metric.reset()


def build_metrics(config, mode):
    if mode == 'train' and "Metric" in config and "Train" in config[
            "Metric"] and config["Metric"]["Train"]:
        metric_config = config["Metric"]["Train"]
        if config["DataLoader"]["Train"]["dataset"].get("batch_transform_ops",
                                                        None):
            for m_idx, m in enumerate(metric_config):
                if "TopkAcc" in m:
                    msg = f"Unable to calculate accuracy when using \"batch_transform_ops\". The metric \"{m}\" has been removed."
                    logger.warning(msg)
                    metric_config.pop(m_idx)
        train_metric_func = CombinedMetrics(copy.deepcopy(metric_config))
        return train_metric_func

    if mode == "eval" or (mode == "train" and
                          config["Global"]["eval_during_train"]):
        eval_mode = config["Global"].get("eval_mode", "classification")
        if eval_mode == "classification":
            if "Metric" in config and "Eval" in config["Metric"]:
                eval_metric_func = CombinedMetrics(
                    copy.deepcopy(config["Metric"]["Eval"]))
            else:
                eval_metric_func = None
        elif eval_mode == "retrieval":
            if "Metric" in config and "Eval" in config["Metric"]:
                metric_config = config["Metric"]["Eval"]
            else:
                metric_config = [{"name": "Recallk", "topk": (1, 5)}]
            eval_metric_func = CombinedMetrics(copy.deepcopy(metric_config))
        return eval_metric_func
