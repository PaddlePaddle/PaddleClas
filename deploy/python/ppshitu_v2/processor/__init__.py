from .postprocessor import build_postprocessor
from .preprocessor import build_preprocessor
from .predictor import build_predictor
from .searcher import build_searcher
from .router import build_router

from .base_processor import BaseProcessor


class AlgoMod:
    def __init__(self, config):
        super().__init__(config)
        self.processors = []
        for processor_config in config["processors"]:
            processor_type = processor_config.get("type")

            if processor_type == "preprocessor":
                processor = build_preprocessor(processor_config)
            elif processor_type == "predictor":
                processor = build_predictor(processor_config)
            elif processor_type == "postprocessor":
                processor = build_postprocessor(processor_config)
            elif processor_type == "searcher":
                processor = build_searcher(processor_config)
            else:
                raise NotImplemented("processor type {} unknown.".format(
                    processor_type))
            self.processors.append(processor)

        if config.get("router", None) is not None:
            router_processor = build_router(config["router_config"], self.processors)
            self.processors = [router_processor]

    def process(self, input_data):
        for processor in self.processors:
            input_data = processor.process(input_data)
        return input_data
