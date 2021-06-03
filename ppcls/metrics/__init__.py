import paddle
import paddle.nn as nn
import copy

class Topk(nn.Layer):
    def __init__(self, topk=[1, 5]):
        super().__init__()
        assert isinstance(topk, (int, list))
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

def build_metric(config):
    config = copy.deepcopy(config)
    metrics_func = Topk()
    logger.info("build metric {} success.".format(metrics_func))
    return metrics_func
