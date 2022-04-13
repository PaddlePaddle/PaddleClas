import importlib

from processor.postprocessor.cls import TopK
from processor.postprocessor.det import DetPostPro
from processor.postprocessor.rec import FeatureNormalizer


def build_postprocessor(config):
    processor_mod = importlib.import_module(__name__)
    processor_name = config.get("name")
    return getattr(processor_mod, processor_name)(config)
