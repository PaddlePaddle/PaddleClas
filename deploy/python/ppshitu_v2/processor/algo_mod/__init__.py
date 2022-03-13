# from .postprocessor import build_postprocessor
# from .preprocessor import build_preprocessor
# from .predictor import build_predictor

import importlib

from processor.algo_mod import preprocessor
from processor.algo_mod import predictor
from processor.algo_mod import postprocessor
from processor.algo_mod import searcher

from ..base_processor import BaseProcessor


class AlgoMod(BaseProcessor):
    def __init__(self, config):
        self.processors = []
        for processor_config in config["processors"]:
            processor_type = processor_config.get("type")
            processor_name = processor_config.get("name")
            _mod = importlib.import_module(__name__)
            processor = getattr(
                getattr(_mod, processor_type),
                processor_name)(processor_config)

            # if processor_type == "preprocessor":
            #     processor = build_preprocessor(processor_config)
            # elif processor_type == "predictor":
            #     processor = build_predictor(processor_config)
            # elif processor_type == "postprocessor":
            #     processor = build_postprocessor(processor_config)
            # else:
            #     raise NotImplemented("processor type {} unknown.".format(processor_type))
            self.processors.append(processor)

    def process(self, input_data):
        for processor in self.processors:
            input_data = processor.process(input_data)
        return input_data
