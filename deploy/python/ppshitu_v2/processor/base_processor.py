from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def process(self, input_data):
        pass
