import os
import sys

import cv2

from engine import build_engine
from utils import config


def main():
    args = config.parse_args()
    config_dict = config.get_config(
        args.config, overrides=args.override, show=False)
    config_dict.profiler_options = args.profiler_options
    engine = build_engine(config_dict)
    image_file = "../../images/wangzai.jpg"
    img = cv2.imread(image_file)[:, :, ::-1]
    input_data = {"input_image": img}
    data = engine.process(input_data)

    # for cls
    if "cls_result" in data:
        print(data["cls_result"])
    # for det
    if "det_result" in data:
        print(data["det_result"])
    # for rec
    if "features" in data:
        print(data["features"])


if __name__ == '__main__':
    main()
