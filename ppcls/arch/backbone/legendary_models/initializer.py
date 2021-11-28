from paddle.fluid.initializer import Initializer, TruncatedNormal, Constant, Normal, Xavier
from paddle.fluid import ParamAttr
import numpy as np
import math


class Conv2DWeightInitializer(Initializer):
    def __init__(self, mean=0.0, seed=0):
        super(Conv2DWeightInitializer, self).__init__()
        self.mean = mean
        self.seed = seed

    def __call__(self, param, block=None):
        shape = param.shape
        assert len(shape) == 4, "parameter {} shape {}".format(param.name,
                                                               shape)
        fan_in = np.prod(shape[1:])
        scale = math.sqrt(1.0 / fan_in)
        norm = TruncatedNormal(loc=self.mean, scale=scale, seed=self.seed)
        norm(param, block)


def get_param_attr(name, init_type):
    if init_type == "conv_weight":
        initializer = Conv2DWeightInitializer(mean=0.0)
        # initializer = Xavier(uniform=False, fan_in=None, fan_out=0)
    elif init_type == "bias":
        initializer = Constant(value=0.0)
    elif init_type == "scale":
        initializer = Constant(value=1.0)
    elif init_type == "fc_weight":
        initializer = Normal(loc=0.0, scale=0.01)
    else:
        raise ValueError(
            "init_type must be inside ['conv_weight', 'bias', 'scale', 'fc_weight']"
        )
    return ParamAttr(name=name, initializer=initializer)
