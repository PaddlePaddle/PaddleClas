from processor.base_processor import BaseProcessor


class GetData(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.data_keys = config.get("data_keys")

    def process(self, input_data):
        for i, data_key in enumerate(self.data_keys):
            input_data[self.output_keys[i]] = input_data[self.input_keys[i]][data_key]
        return input_data
