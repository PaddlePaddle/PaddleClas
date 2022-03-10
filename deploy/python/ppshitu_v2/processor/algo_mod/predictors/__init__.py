from processor.algo_mod.predictors.paddle_predictor import Predictor as paddle_predictor
from processor.algo_mod.predictors.onnx_predictor import Predictor as onnx_predictor


def build_predictor(config):
    # if use paddle backend
    if True:
        return paddle_predictor(config)
    # if use onnx backend
    else:
        return onnx_predictor(config)