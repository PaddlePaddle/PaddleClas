from .fake_cls import FakeClassifier


def build_algo_mod(config):
    algo_name = config.get("algo_name")
    if algo_name == "fake_clas":
        return FakeClassifier(config)
