import os
import pickle

import faiss


class Searcher:
    def __init__(self, config):
        super().__init__()

        self.Searcher = faiss.read_index(
            os.path.join(config["index_dir"], "vector.index"))

        with open(os.path.join(config["index_dir"], "id_map.pkl"), "rb") as fd:
            self.id_map = pickle.load(fd)

        self.return_k = config["return_k"]

    def process(self, data):
        features = data["features"]
        scores, docs = self.Searcher.search(features, self.return_k)
        data["search_res"] = (scores, docs)
        return data
