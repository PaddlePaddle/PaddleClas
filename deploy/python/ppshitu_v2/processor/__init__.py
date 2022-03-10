from abc import ABC, abstractmethod

from processor.algo_mod import searcher
from processor.algo_mod.predictors import build_predictor

# def build_processor(config):
#     print(config)
#     processor_type = config.get("processor_type")
#     processor_mod = locals()[processor_type]
#     processor_name = config.get("processor_name")
#     return getattr(processor_mod, processor_name)

# class BaseProcessor(ABC):
#     @abstractmethod
#     def __init__(self, config):
#         pass

#     @abstractmethod
#     def process(self, input_data):
#         pass
