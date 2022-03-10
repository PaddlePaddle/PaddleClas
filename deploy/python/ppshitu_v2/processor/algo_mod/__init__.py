from .. import BaseProcessor, build_processor


class AlgoMod(BaseProcessor):
    def __init__(self, config):
        self.pre_processor = build_processor(config["pre_processor"])
        self.predictor = build_processor(config["predictor"])
        self.post_processor = build_processor(config["post_processor"])

    def process(self, input_data):
        input_data = self.pre_processor(input_data)
        input_data = self.predictor(input_data)
        input_data = self.post_processor(input_data)
        return input_data
