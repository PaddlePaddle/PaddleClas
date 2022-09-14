import os

from paddleclas.deploy.utils import logger
from paddleclas.deploy.utils import config
from paddleclas.deploy.utils.predictor import Predictor
from paddleclas.deploy.utils.get_image_list import get_image_list
from paddleclas.deploy.python.det_preprocess import det_preprocess
from paddleclas.deploy.python.predict_det import DetPredictor
from paddleclas.deploy.python.predict_rec import RecPredictor
from paddleclas.deploy.python.preprocess import create_operators
from get_images_list_from_txt import get_image_list_from_txt

import os
import time
import yaml
import ast
from functools import reduce
import cv2
import numpy as np

from generate_index import random_sample, sample_all


def main(config):
    datasets_list = config['Datasets']
    assert datasets_list, "Datasets not found ..."

    assert config["Global"]["batch_size"] == 1

    for dataset in datasets_list:
        print("\nSampling indexes of %s" % (dataset))
        infer_path = datasets_list[dataset]["infer_path"]
        infer_imgs = datasets_list[dataset]["infer_imgs"]

        images_list = get_image_list_from_txt(dataset, infer_imgs, infer_path)
        if images_list == None or len(images_list) == 0:
            print("not found any img file in {}\n".format(dataset))
            continue

        output_dir = datasets_list[dataset]["output_dir"]
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)

        for method in config['Methods']:
            if config['Methods'][method]['method_name'] == 'RandomSample':
                gallery_num = config['Methods'][method]["gallery_num"]
                random_sample(config, method, images_list, gallery_num,
                              output_dir)

            if config['Methods'][method]['method_name'] == 'SampleAll':
                sample_all(config, method, images_list, output_dir)


# build index from gallery images
# python shitu_index_selector/sample_indexes.py
#        -c configs/sample_indexes.yaml 
if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
