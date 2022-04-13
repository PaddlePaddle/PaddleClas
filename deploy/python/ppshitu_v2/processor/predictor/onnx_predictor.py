from processor.base_processor import BaseProcessor


class ONNXPredictor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)

    def process(self, input_data):
        raise NotImplemented("ONNXPredictor Not supported yet")
