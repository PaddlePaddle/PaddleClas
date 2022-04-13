import numpy as np

from processor.base_processor import BaseProcessor


class FeatureNormalizer(BaseProcessor):
    def __init__(self, config=None):
        super().__init__(config)
        if self.input_keys is None:
            self.input_keys = ["features"]
        if self.output_keys is None:
            self.output_keys = ["features"]

    def process(self, input_data):
        batch_output = input_data[self.input_keys[0]]
        feas_norm = np.sqrt(
            np.sum(np.square(batch_output), axis=1, keepdims=True))
        batch_output = np.divide(batch_output, feas_norm)
        input_data[self.output_keys[0]] = batch_output
        return input_data
