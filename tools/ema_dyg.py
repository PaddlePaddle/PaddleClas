import paddle
import numpy as np


class ExponentialMovingAverage():
    def __init__(self, model, decay, thres_steps=True):
        self._model = model
        self._decay = decay
        self._thres_steps = thres_steps
        self._shadow = {}
        self._backup = {}

    def register(self):
        self._update_step = 0
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                self._shadow[name] = param.numpy().copy()

    def update(self):
        decay = min(self._decay, (1 + self._update_step) / (
            10 + self._update_step)) if self._thres_steps else self._decay
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                assert name in self._shadow
                new_val = np.array(param.numpy().copy())
                old_val = np.array(self._shadow[name])
                new_average = decay * old_val + (1 - decay) * new_val
                self._shadow[name] = new_average
        self._update_step += 1
        return decay

    def apply(self):
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                assert name in self._shadow
                self._backup[name] = np.array(param.numpy().copy())
                param.set_value(np.array(self._shadow[name]))

    def restore(self):
        for name, param in self._model.named_parameters():
            if param.stop_gradient is False:
                assert name in self._backup
                param.set_value(self._backup[name])
        self._backup = {}
