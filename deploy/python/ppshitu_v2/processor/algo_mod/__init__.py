from processor.algo_mod.data_processor import ImageProcessor
from processor.algo_mod.post_processor.det import DetPostProcessor
from processor.algo_mod.predictors import build_predictor


def build_processor(config):
    # processor_type = config.get("processor_type")
    # processor_mod = locals()[processor_type]
    processor_name = config.get("name")
    return eval(processor_name)(config)


class AlgoMod(object):
    def __init__(self, config):
        self.pre_processor = build_processor(config["preprocess"])
        self.predictor = build_predictor(config["predictor"])
        self.post_processor = build_processor(config["postprocess"])

    def process(self, input_data):
        input_data = self.pre_processor.process(input_data)
        input_data = self.predictor.process(input_data)
        input_data = self.post_processor.process(input_data)
        return input_data
