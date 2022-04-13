import importlib

from processor.preprocessor.image_processor import ImageProcessor
from processor.preprocessor.basic_processor import GetData


def build_preprocessor(config):
    processor_mod = importlib.import_module(__name__)
    processor_name = config.get("name")
    return getattr(processor_mod, processor_name)(config)
