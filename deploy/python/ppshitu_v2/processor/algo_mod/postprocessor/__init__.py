import importlib

from .det import PPYOLOv2PostPro


def build_postprocessor(config):
    processor_mod = importlib.import_module(__name__)
    processor_name = config.get("name")
    return getattr(processor_mod, processor_name)(config)
