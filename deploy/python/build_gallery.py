# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pickle

import cv2
import faiss
import numpy as np
from paddleclas.deploy.python.predict_rec import RecPredictor
from paddleclas.deploy.utils import config, logger
from tqdm import tqdm


def split_datafile(data_file, image_root, delimiter="\t"):
    '''
        data_file: image path and info, which can be splitted by spacer
        image_root: image path root
        delimiter: delimiter
    '''
    gallery_images = []
    gallery_docs = []
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _, ori_line in enumerate(lines):
            line = ori_line.strip().split(delimiter)
            text_num = len(line)
            assert text_num >= 2, f"line({ori_line}) must be splitted into at least 2 parts, but got {text_num}"
            image_file = os.path.join(image_root, line[0])

            gallery_images.append(image_file)
            gallery_docs.append(ori_line.strip())

    return gallery_images, gallery_docs


class GalleryBuilder(object):
    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)
        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.android_demo = config["Global"].get("android_demo", False)
        self.build(config['IndexProcess'])

    def build(self, config):
        '''
            build index from scratch
        '''
        operation_method = config.get("index_operation", "new").lower()

        gallery_images, gallery_docs = split_datafile(
            config['data_file'], config['image_root'], config['delimiter'])

        # when remove data in index, do not need extract fatures
        if operation_method != "remove":
            gallery_features = self._extract_features(gallery_images, config)
        assert operation_method in [
            "new", "remove", "append"
        ], "Only append, remove and new operation are supported"

        if self.android_demo:
            self._create_index_for_android_demo(config, gallery_features, gallery_docs)
            return

        # vector.index: faiss index file
        # id_map.pkl: use this file to map id to image_doc
        index, ids = None, None
        if operation_method in ["remove", "append"]:
            # if remove or append, load vector.index and id_map.pkl
            index, ids = self._load_index(config)
            index_method = config.get("index_method", "HNSW32")
        else:
            index_method, index, ids = self._create_index(config)
        if index_method == "HNSW32":
            logger.warning(
                "The HNSW32 method dose not support 'remove' operation")

        if operation_method != "remove":
            # calculate id for new data
            index, ids = self._add_gallery(index, ids, gallery_features, gallery_docs, config, operation_method)
        else:
            if index_method == "HNSW32":
                raise RuntimeError(
                    "The index_method: HNSW32 dose not support 'remove' operation"
                )
            # remove ids in id_map, remove index data in faiss index
            index, ids = self._rm_id_in_galllery(index, ids, gallery_docs)

        # store faiss index file and id_map file
        self._save_gallery(config, index, ids)

    def _create_index_for_android_demo(self, config, gallery_features, gallery_docs):
        if not os.path.exists(config["index_dir"]):
            os.makedirs(config["index_dir"], exist_ok=True)
        #build index
        index = faiss.IndexFlatIP(config["embedding_size"])
        index.add(gallery_features)

        # calculate id for data
        ids_now = (np.arange(0, len(gallery_docs))).astype(np.int64)
        ids = {}
        for i, d in zip(list(ids_now), gallery_docs):
            ids[i] = d
        self._save_gallery(config, index, ids)

    def _extract_features(self, gallery_images, config):
        # extract gallery features
        if config["dist_type"] == "hamming":
            gallery_features = np.zeros(
                [len(gallery_images), config['embedding_size'] // 8],
                dtype=np.uint8)
        else:
            gallery_features = np.zeros(
                [len(gallery_images), config['embedding_size']],
                dtype=np.float32)

        #construct batch imgs and do inference
        batch_size = config.get("batch_size", 32)
        batch_img = []
        for i, image_file in enumerate(tqdm(gallery_images)):
            img = cv2.imread(image_file)
            if img is None:
                logger.error("img empty, please check {}".format(image_file))
                exit()
            img = img[:, :, ::-1]
            batch_img.append(img)

            if (i + 1) % batch_size == 0:
                rec_feat = self.rec_predictor.predict(batch_img)
                gallery_features[i - batch_size + 1:i + 1, :] = rec_feat
                batch_img = []

        if len(batch_img) > 0:
            rec_feat = self.rec_predictor.predict(batch_img)
            gallery_features[-len(batch_img):, :] = rec_feat
            batch_img = []

        return gallery_features

    def _load_index(self, config):
        assert os.path.join(
            config["index_dir"], "vector.index"
        ), "The vector.index dose not exist in {} when 'index_operation' is not None".format(
            config["index_dir"])
        assert os.path.join(
            config["index_dir"], "id_map.pkl"
        ), "The id_map.pkl dose not exist in {} when 'index_operation' is not None".format(
            config["index_dir"])
        index = faiss.read_index(
            os.path.join(config["index_dir"], "vector.index"))
        with open(os.path.join(config["index_dir"], "id_map.pkl"),
                  'rb') as fd:
            ids = pickle.load(fd)
        assert index.ntotal == len(ids.keys(
        )), "data number in index is not equal in in id_map"
        return index, ids

    def _create_index(self, config):
        if not os.path.exists(config["index_dir"]):
            os.makedirs(config["index_dir"], exist_ok=True)
        index_method = config.get("index_method", "HNSW32")

        # if IVF method, cal ivf number automaticlly
        if index_method == "IVF":
            index_method = index_method + str(
                min(int(len(gallery_images) // 8), 65536)) + ",Flat"

        # for binary index, add B at head of index_method
        if config["dist_type"] == "hamming":
            index_method = "B" + index_method

        #dist_type
        dist_type = faiss.METRIC_INNER_PRODUCT if config[
            "dist_type"] == "IP" else faiss.METRIC_L2

        #build index
        if config["dist_type"] == "hamming":
            index = faiss.index_binary_factory(config["embedding_size"],
                                               index_method)
        else:
            index = faiss.index_factory(config["embedding_size"],
                                        index_method, dist_type)
            index = faiss.IndexIDMap2(index)
        ids = {}
        return index_method, index, ids

    def _add_gallery(self, index, ids, gallery_features, gallery_docs, config, operation_method):
        start_id = max(ids.keys()) + 1 if ids else 0
        ids_now = (
            np.arange(0, len(gallery_docs)) + start_id).astype(np.int64)

        # only train when new index file
        if operation_method == "new":
            if config["dist_type"] == "hamming":
                index.add(gallery_features)
            else:
                index.train(gallery_features)

        if not config["dist_type"] == "hamming":
            index.add_with_ids(gallery_features, ids_now)

        for i, d in zip(list(ids_now), gallery_docs):
            ids[i] = d
        return index, ids

    def _rm_id_in_galllery(self, index, ids, gallery_docs):
        remove_ids = list(
            filter(lambda k: ids.get(k) in gallery_docs, ids.keys()))
        remove_ids = np.asarray(remove_ids)
        index.remove_ids(remove_ids)
        for k in remove_ids:
            del ids[k]

        return index, ids

    def _save_gallery(self, config, index, ids):
        if config["dist_type"] == "hamming":
            faiss.write_index_binary(
                index, os.path.join(config["index_dir"], "vector.index"))
        else:
            faiss.write_index(
                index, os.path.join(config["index_dir"], "vector.index"))

        with open(os.path.join(config["index_dir"], "id_map.pkl"), 'wb') as fd:
            pickle.dump(ids, fd)


def main(config):
    GalleryBuilder(config)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
