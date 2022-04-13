import os
import pickle

import faiss


def build_searcher(config):
    return Searcher(config)


class Searcher:
    def __init__(self, config):
        super().__init__()

        self.faiss_searcher = faiss.read_index(
            os.path.join(config["index_dir"], "vector.index"))

        with open(os.path.join(config["index_dir"], "id_map.pkl"), "rb") as fd:
            self.id_map = pickle.load(fd)

        self.return_k = config["return_k"]

    def process(self, data):
        features = data["features"]
        scores, docs = self.faiss_searcher.search(features, self.return_k)

        preds = {}
        preds["rec_docs"] = self.id_map[docs[0][0]].split()[1]
        preds["rec_scores"] = scores[0][0]

        data["search_res"] = preds
        return data
