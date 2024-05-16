import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ["AdaptiveAvgPool2D"]


class AdaptiveAvgPool2D(nn.AdaptiveAvgPool2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if paddle.device.get_device().startswith("npu"):
            self.device = "npu"
        else:
            self.device = None

        if isinstance(self._output_size, int) and self._output_size == 1:
            self._gap = True
        elif isinstance(self._output_size, tuple) and self._output_size[
                0] == 1 and self._output_size[1] == 1:
            self._gap = True
        else:
            self._gap = False

    def forward(self, x):
        if self.device == "npu" and self._gap:
            # Global Average Pooling
            N, C, _, _ = x.shape
            x_mean = paddle.mean(x, axis=[2, 3])
            x_mean = paddle.reshape(x_mean, [N, C, 1, 1])
            return x_mean
        else:
            return F.adaptive_avg_pool2d(
                x,
                output_size=self._output_size,
                data_format=self._data_format,
                name=self._name, )
