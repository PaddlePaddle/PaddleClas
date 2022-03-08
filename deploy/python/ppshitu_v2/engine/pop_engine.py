from ..processor import build_processor


class POPEngine:
    def __init__(self, config):
        self.processor_list = []
        last_algo_type = "start"
        for processor_config in config["Processors"]:
            processor_config["last_algo_type"] = last_algo_type
            self.processor_list.append(build_processor(processor_config))
            last_algo_type = processor_config["type"]

    def process(self, x):
        for processor in self.processor_list:
            x = processor.process(x)
        return x
