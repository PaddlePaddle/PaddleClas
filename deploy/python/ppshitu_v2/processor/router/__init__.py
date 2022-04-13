import importlib

from processor.router.loop_router import LoopRouter


def build_router(config, processors):
    processor_mod = importlib.import_module(__name__)
    processor_name = config.pop("router")
    return getattr(processor_mod, processor_name)(config, processors)