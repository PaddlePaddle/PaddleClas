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

from python.predict_rec import RecPredictor
from python.predict_det import DetPredictor
from vector_search import Graph_Index

from utils import logger
from utils import config
from utils.get_image_list import get_image_list


def split_datafile(data_file, image_root):
    gallery_images = []
    gallery_docs = []
    with open(data_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split("\t")
            if line[0] == 'image_id':
                 continue
            image_file = os.path.join(image_root, line[3])
            image_doc = line[1]
            gallery_images.append(image_file)
            gallery_docs.append(image_doc)
    return gallery_images, gallery_docs


class SystemPredictor(object):
    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)
        self.det_predictor = DetPredictor(config)

        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.indexer(config['IndexProcess'])
        self.return_k = self.config['IndexProcess']['infer']['return_k']
        self.search_budget = self.config['IndexProcess']['infer']['search_budget']
    
    def indexer(self, config):
        if 'build' in config.keys() and config['build']['enable']:  # build the index from scratch    
            with open(config['build']['data_file']) as f:
                lines = f.readlines()
            gallery_images, gallery_docs = split_datafile(config['build']['data_file'], config['build']['image_root'])
            # extract gallery features
            gallery_features = np.zeros([len(gallery_images), config['build']['embedding_size']], dtype=np.float32)
            for i, image_file in enumerate(gallery_images):
                img = cv2.imread(image_file)[:, :, ::-1]
                rec_feat = self.rec_predictor.predict(img)
                gallery_features[i,:] = rec_feat
            # train index 
            self.Searcher = Graph_Index(dist_type=config['build']['dist_type']) 
            self.Searcher.build(gallery_vectors=gallery_features, gallery_docs=gallery_docs, 
                pq_size=config['build']['pq_size'], index_path=config['build']['index_path'])
            
        else:   # load local index
            self.Searcher = Graph_Index(dist_type=config['build']['dist_type']) 
            self.Searcher.load(config['infer']['index_path'])

    def predict(self, img):
        output = []
        results = self.det_predictor.predict(img)
        for result in results:
            xmin, ymin, xmax, ymax = result["bbox"].astype("int")
            crop_img = img[xmin:xmax, ymin:ymax, :].copy()
            rec_results = self.rec_predictor.predict(crop_img)
            result["feature"] = rec_results

            scores, docs = self.Searcher.search(query=rec_results, return_k=self.return_k, search_budget=self.search_budget)
            result["ret_docs"] = docs
            result["ret_scores"] = scores

            output.append(result)
        return output
            


def main(config):
    system_predictor = SystemPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    assert config["Global"]["batch_size"] == 1
    for idx, image_file in enumerate(image_list):
        img = cv2.imread(image_file)[:, :, ::-1]
        output = system_predictor.predict(img)
        #print(output)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
