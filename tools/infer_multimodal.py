from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config
from ppcls.engine.engine_multimodal import EngineMultimodal

if __name__ == "__main__":
    args = config.parse_args()
    args.config = "./ppcls/configs/ram/config_train.yaml"
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    engine = EngineMultimodal(config, mode="infer")
    engine.infer()
