import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import cv2

from engine import build_engine
from utils import config
from utils.get_image_list import get_image_list

import numpy as np


def load_vector(path):
    return np.load(path)


def main():
    args = config.parse_args()
    config_dict = config.get_config(
        args.config, overrides=args.override, show=False)
    config_dict.profiler_options = args.profiler_options
    engine = build_engine(config_dict)
    vector = load_vector(config_dict["Global"]["infer_imgs"])
    output = engine.process({"features": vector})
    print(output["search_res"])


if __name__ == '__main__':
    main()
