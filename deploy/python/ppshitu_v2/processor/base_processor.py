from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    @abstractmethod
    def __init__(self, config):
        self.input_keys = config.get("input_keys", None)
        self.output_keys = config.get("output_keys", None)

    @abstractmethod
    def process(self, input_data):
        pass
