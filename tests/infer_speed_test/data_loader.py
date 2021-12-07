import sys
import numpy as np


def get_dataloader(config_dict):
    data_loader = sys.modules[__name__]
    return getattr(data_loader, config_dict["method"])


class NumpyGenerator(object):
    def __init__(self, config_dict):
        self.gen_method = config_dict["Data"]["gen_method"]
        self.input_shape = config_dict["Data"]["data_shape"]
        batchsize = config_dict["Variables"]["batchsize"]
        self.input_shape.insert(0, batchsize)

    def gen_data(self):
        if self.gen_method == "uniform":
            return np.random.uniform(size=self.input_shape)
        elif self.gen_method == "ones":
            return np.ones(shape=self.input_shape)
        elif self.gen_method == "zeros":
            return np.zeros(shape=self.input_shape)
