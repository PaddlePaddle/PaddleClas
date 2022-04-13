import importlib

from processor import AlgoMod


class POPEngine:
    def __init__(self, config):
        self.algo_list = []
        current_mod = importlib.import_module(__name__)
        for mod_config in config["Modules"]:
            mod = AlgoMod(mod_config)
            self.algo_list.append(mod)

    def process(self, input_data):
        for algo_module in self.algo_list:
            input_data = algo_module.process(input_data)
        return input_data
