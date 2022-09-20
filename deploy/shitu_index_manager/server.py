# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import mod.mainwindow

from paddleclas.deploy.utils import config, logger
from paddleclas.deploy.python.predict_rec import RecPredictor
from fastapi import FastAPI
import uvicorn
import numpy as np
import faiss
from typing import List
import pickle
import cv2
import socket
import json
import operator
from multiprocessing import Process
"""
完整的index库如下:
root_path/            # 库存储目录
|-- image_list.txt     # 图像列表，每行：image_path label。由前端生成及修改。后端只读
|-- features.pkl       # 建库之后，保存的embedding向量，后端生成，前端无需操作
|-- images             # 图像存储目录，由前端生成及增删查等操作。后端只读
|   |-- md5.jpg
|   |-- md5.jpg
|   |-- ……
|-- index              # 真正的生成的index库存储目录，后端生成及操作，前端无需操作。
|   |-- vector.index   # faiss生成的索引库
|   |-- id_map.pkl     # 索引文件
"""


class ShiTuIndexManager(object):
    def __init__(self, config):
        self.root_path = None
        self.image_list_path = "image_list.txt"
        self.image_dir = "images"
        self.index_path = "index/vector.index"
        self.id_map_path = "index/id_map.pkl"
        self.features_path = "features.pkl"
        self.index = None
        self.id_map = None
        self.features = None
        self.config = config
        self.predictor = RecPredictor(config)

    def _load_pickle(self, path):
        if os.path.exists(path):
            return pickle.load(open(path, 'rb'))
        else:
            return None

    def _save_pickle(self, path, data):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fd:
            pickle.dump(data, fd)

    def _load_index(self):
        self.index = faiss.read_index(
            os.path.join(self.root_path, self.index_path))
        self.id_map = self._load_pickle(
            os.path.join(self.root_path, self.id_map_path))
        self.features = self._load_pickle(
            os.path.join(self.root_path, self.features_path))

    def _save_index(self, index, id_map, features):
        faiss.write_index(index, os.path.join(self.root_path, self.index_path))
        self._save_pickle(
            os.path.join(self.root_path, self.id_map_path), id_map)
        self._save_pickle(
            os.path.join(self.root_path, self.features_path), features)

    def _update_path(self, root_path, image_list_path=None):
        if root_path == self.root_path:
            pass
        else:
            self.root_path = root_path
            if not os.path.exists(os.path.join(root_path, "index")):
                os.mkdir(os.path.join(root_path, "index"))
            if image_list_path is not None:
                self.image_list_path = image_list_path

    def _cal_featrue(self, image_list):
        batch_images = []
        featrures = None
        cnt = 0
        for idx, image_path in enumerate(image_list):
            image = cv2.imread(image_path)
            if image is None:
                return "{} is broken or not exist. Stop"
            else:
                image = image[:, :, ::-1]
                batch_images.append(image)
                cnt += 1
            if cnt % self.config["Global"]["batch_size"] == 0 or (
                    idx + 1) == len(image_list):
                if len(batch_images) == 0:
                    continue
                batch_results = self.predictor.predict(batch_images)
                featrures = batch_results if featrures is None else np.concatenate(
                    (featrures, batch_results), axis=0)
                batch_images = []
        return featrures

    def _split_datafile(self, data_file, image_root):
        '''
        data_file: image path and info, which can be splitted by spacer
        image_root: image path root
        delimiter: delimiter
        '''
        gallery_images = []
        gallery_docs = []
        gallery_ids = []
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for _, ori_line in enumerate(lines):
                line = ori_line.strip().split()
                text_num = len(line)
                assert text_num >= 2, f"line({ori_line}) must be splitted into at least 2 parts, but got {text_num}"
                image_file = os.path.join(image_root, line[0])

                gallery_images.append(image_file)
                gallery_docs.append(ori_line.strip())
                gallery_ids.append(os.path.basename(line[0]).split(".")[0])

        return gallery_images, gallery_docs, gallery_ids

    def create_index(self,
                     image_list: str,
                     index_method: str="HNSW32",
                     image_root: str=None):
        if not os.path.exists(image_list):
            return "{} is not exist".format(image_list)
        if index_method.lower() not in ['hnsw32', 'ivf', 'flat']:
            return "The index method Only support: HNSW32, IVF, Flat"
        self._update_path(os.path.dirname(image_list), image_list)

        # get image_paths
        image_root = image_root if image_root is not None else self.root_path
        gallery_images, gallery_docs, image_ids = self._split_datafile(
            image_list, image_root)

        # gernerate index
        if index_method == "IVF":
            index_method = index_method + str(
                min(max(int(len(gallery_images) // 32), 2), 65536)) + ",Flat"
        index = faiss.index_factory(
            self.config["IndexProcess"]["embedding_size"], index_method,
            faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap2(index)
        features = self._cal_featrue(gallery_images)
        self.index.train(features)
        index_ids = np.arange(0, len(gallery_images)).astype(np.int64)
        self.index.add_with_ids(features, index_ids)

        self.id_map = dict()
        for i, d in zip(list(index_ids), gallery_docs):
            self.id_map[i] = d

        self.features = {
            "features": features,
            "index_method": index_method,
            "image_ids": image_ids,
            "index_ids": index_ids.tolist()
        }
        self._save_index(self.index, self.id_map, self.features)

    def open_index(self, root_path: str, image_list_path: str) -> str:
        self._update_path(root_path)
        _, _, image_ids = self._split_datafile(image_list_path, root_path)
        if os.path.exists(os.path.join(self.root_path, self.index_path)) and \
                os.path.exists(os.path.join(self.root_path, self.id_map_path)) and \
                os.path.exists(os.path.join(self.root_path, self.features_path)):
            self._update_path(root_path)
            self._load_index()
            if operator.eq(set(image_ids), set(self.features['image_ids'])):
                return ""
            else:
                return "The image list is different from index, Please update index"
        else:
            return "File not exist: features.pkl, vector.index, id_map.pkl"

    def update_index(self, image_list: str, image_root: str=None) -> str:
        if self.index and self.id_map and self.features:
            image_paths, image_docs, image_ids = self._split_datafile(
                image_list, image_root
                if image_root is not None else self.root_path)

            # for add image
            add_ids = list(
                set(image_ids).difference(set(self.features["image_ids"])))
            add_indexes = [i for i, x in enumerate(image_ids) if x in add_ids]
            add_image_paths = [image_paths[i] for i in add_indexes]
            add_image_docs = [image_docs[i] for i in add_indexes]
            add_image_ids = [image_ids[i] for i in add_indexes]
            self._add_index(add_image_paths, add_image_docs, add_image_ids)

            # delete images
            delete_ids = list(
                set(self.features["image_ids"]).difference(set(image_ids)))
            self._delete_index(delete_ids)
            self._save_index(self.index, self.id_map, self.features)
            return ""
        else:
            return "Failed. Please create or open index first"

    def _add_index(self, image_list: List, image_docs: List, image_ids: List):
        if len(image_ids) == 0:
            return
        featrures = self._cal_featrue(image_list)
        index_ids = (
            np.arange(0, len(image_list)) + max(self.id_map.keys()) + 1
        ).astype(np.int64)
        self.index.add_with_ids(featrures, index_ids)

        for i, d in zip(index_ids, image_docs):
            self.id_map[i] = d

        self.features['features'] = np.concatenate(
            [self.features['features'], featrures], axis=0)
        self.features['image_ids'].extend(image_ids)
        self.features['index_ids'].extend(index_ids.tolist())

    def _delete_index(self, image_ids: List):
        if len(image_ids) == 0:
            return
        indexes = [
            i for i, x in enumerate(self.features['image_ids'])
            if x in image_ids
        ]
        self.features["features"] = np.delete(
            self.features["features"], indexes, axis=0)
        self.features["image_ids"] = np.delete(
            np.asarray(self.features["image_ids"]), indexes, axis=0).tolist()
        index_ids = np.delete(
            np.asarray(self.features["index_ids"]), indexes, axis=0).tolist()
        id_map_values = [self.id_map[i] for i in index_ids]
        self.index.reset()
        ids = np.arange(0, len(id_map_values)).astype(np.int64)
        self.index.add_with_ids(self.features['features'], ids)
        self.id_map.clear()
        for i, d in zip(ids, id_map_values):
            self.id_map[i] = d
        self.features["index_ids"] = ids


app = FastAPI()


@app.get("/new_index")
def new_index(image_list_path: str,
              index_method: str="HNSW32",
              index_root_path: str=None,
              force: bool=False):
    result = ""
    try:
        if index_root_path is not None:
            image_list_path = os.path.join(index_root_path, image_list_path)
        index_path = os.path.join(index_root_path, "index", "vector.index")
        id_map_path = os.path.join(index_root_path, "index", "id_map.pkl")

        if not (os.path.exists(index_path) and
                os.path.exists(id_map_path)) or force:
            manager.create_index(image_list_path, index_method,
                                 index_root_path)
        else:
            result = "There alrealy has index in {}".format(index_root_path)
    except Exception as e:
        result = e.__str__()
    data = {"error_message": result}
    return json.dumps(data).encode()


@app.get("/open_index")
def open_index(index_root_path: str, image_list_path: str):
    result = ""
    try:
        image_list_path = os.path.join(index_root_path, image_list_path)
        result = manager.open_index(index_root_path, image_list_path)
    except Exception as e:
        result = e.__str__()

    data = {"error_message": result}
    return json.dumps(data).encode()


@app.get("/update_index")
def update_index(image_list_path: str, index_root_path: str=None):
    result = ""
    try:
        if index_root_path is not None:
            image_list_path = os.path.join(index_root_path, image_list_path)
        result = manager.update_index(
            image_list=image_list_path, image_root=index_root_path)
    except Exception as e:
        result = e.__str__()
    data = {"error_message": result}
    return json.dumps(data).encode()


def FrontInterface(server_process=None):
    front = QtWidgets.QApplication([])
    main_window = mod.mainwindow.MainWindow(process=server_process)
    main_window.showMaximized()
    sys.exit(front.exec_())


def Server(app, host, port):
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    args = config.parse_args()
    model_config = config.get_config(
        args.config, overrides=args.override, show=True)
    manager = ShiTuIndexManager(model_config)
    ip = model_config.get('ip', None)
    port = model_config.get('port', None)
    if ip is None or port is None:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except:
            ip = '127.0.0.1'
        port = 8000
    Server(app, ip, port)
