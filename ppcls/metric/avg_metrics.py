from paddle import nn


class AvgMetrics(nn.Layer):
    def __init__(self):
        super().__init__()
        self.avg_meters = {}

    def reset(self):
        self.avg_meters = {}

    @property
    def avg(self):
        if self.avg_meters:
            for metric_key in self.avg_meters:
                return self.avg_meters[metric_key].avg

    @property
    def avg_info(self):
        return ", ".join([self.avg_meters[key].avg_info for key in self.avg_meters])
