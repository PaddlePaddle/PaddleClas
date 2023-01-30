import numpy
import paddle


class ScoreOutput(object):
    def __init__(self, decimal_places):
        self.decimal_places = decimal_places

    def __call__(self, x, file_names=None):
        paddle.set_printoptions(precision=self.decimal_places)
        y = []
        for idx, probs in enumerate(x):
            score = x[0]
            result = {"scores": score}
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y