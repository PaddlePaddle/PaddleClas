import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

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
    if "classification_res" in data:
        print(data["classification_res"])
    # for det
    elif "detection_res" in data:
        print(data["detection_res"])
    # for rec
    elif "features" in data["pred"]:
        features = data["pred"]["features"]
        print(features)
        print(features.shape)
        print(type(features))
    else:
        print("ERROR")


if __name__ == '__main__':
    main()
