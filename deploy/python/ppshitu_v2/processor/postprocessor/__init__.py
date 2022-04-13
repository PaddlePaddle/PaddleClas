import importlib

from .cls import TopK
from .det import DetPostPro
from .rec import FeatureNormalizer


def build_postprocessor(config):
    processor_mod = importlib.import_module(__name__)
    processor_name = config.get("name")
    return getattr(processor_mod, processor_name)(config)
