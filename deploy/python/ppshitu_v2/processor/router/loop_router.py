from processor.base_processor import BaseProcessor


class LoopRouter(BaseProcessor):
    def __init__(self, config, processors):
        super().__init__(config)
        self.processors = processors

    def process(self, input_data):
        length = len(input_data[self.input_keys[0]])
        for i in range(length):
            input_data_i = input_data.copy()
            for k in self.input_keys:
                input_data_i[k] = input_data[k][i]
            for processor in self.processors:
                processor.process(input_data_i)
            for k in self.output_keys:
                if k not in input_data:
                    input_data[k] = [input_data_i[k]]
                else:
                    input_data[k].append(input_data_i[k])
        return input_data
