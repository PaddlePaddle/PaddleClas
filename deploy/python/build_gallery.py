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
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import copy
import cv2
import numpy as np
from tqdm import tqdm

from python.predict_rec import RecPredictor
from vector_search import Graph_Index

from utils import logger
from utils import config


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

            image_doc = line[1]
            gallery_images.append(image_file)
            gallery_docs.append(image_doc)

    return gallery_images, gallery_docs


class GalleryBuilder(object):
    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)
        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.build(config['IndexProcess'])

    def build(self, config):
        '''
            build index from scratch
        '''
        gallery_images, gallery_docs = split_datafile(
            config['data_file'], config['image_root'], config['delimiter'])

        # extract gallery features
        gallery_features = np.zeros(
            [len(gallery_images), config['embedding_size']], dtype=np.float32)

        for i, image_file in enumerate(tqdm(gallery_images)):
            img = cv2.imread(image_file)
            if img is None:
                logger.error("img empty, please check {}".format(image_file))
                exit()
            img = img[:, :, ::-1]
            rec_feat = self.rec_predictor.predict(img)
            gallery_features[i, :] = rec_feat

        # train index 
        self.Searcher = Graph_Index(dist_type=config['dist_type'])
        self.Searcher.build(
            gallery_vectors=gallery_features,
            gallery_docs=gallery_docs,
            pq_size=config['pq_size'],
            index_path=config['index_path'],
            append_index=config["append_index"])


def main(config):
    system_builder = GalleryBuilder(config)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
