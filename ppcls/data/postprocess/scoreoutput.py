import numpy
import numpy as np
import paddle


class ScoreOutput(object):
    def __init__(self, decimal_places):
        self.decimal_places = decimal_places

    def __call__(self, x, file_names=None):
        y = []
        for idx, probs in enumerate(x):
            score = np.around(x[idx].numpy(), self.decimal_places)
            result = {"scores": score}
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y