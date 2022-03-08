from abc import ABC, abstractmethod

from algo_mod import build_algo_mod
from searcher import build_searcher
from data_processor import build_data_processor


def build_processor(config):
    processor_type = config.get("processor_type")
    if processor_type == "algo_mod":
        return build_algo_mod(config)
    elif processor_type == "searcher":
        return build_searcher(config)
    elif processor_type == "data_processor":
        return build_data_processor(config)
    else:
        raise NotImplemented("processor_type {} not implemented.".format(processor_type))


class BaseProcessor(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def process(self, input_data):
        pass
