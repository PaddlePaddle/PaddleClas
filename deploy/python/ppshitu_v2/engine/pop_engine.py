from processor.algo_mod import AlgoMod


class POPEngine:
    def __init__(self, config):
        self.algo_list = []
        # last_algo_type = "start"
        for algo_config in config["AlgoModule"]:
            # algo_config["last_algo_type"] = last_algo_type
            self.algo_list.append(AlgoMod(algo_config["Module"]))
            # last_algo_type = algo_config["type"]

    def process(self, x):
        for algo_module in self.algo_list:
            x = algo_module.process(x)
        return x
