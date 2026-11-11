# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import yaml

import tarfile  # noqa: F401

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from ppcls.engine.engine import Engine
from ppcls.utils import config, convert_to_dict

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)
    cfg.profiler_options = args.profiler_options

    # TIPC expects checkpoints and train.log under:
    #   ${Global.output_dir}/${Arch.name}/
    cfg["Global"]["output_dir"] = os.path.join(
        cfg["Global"]["output_dir"], cfg["Arch"]["name"]
    )

    uniform_output_enabled = cfg["Global"].get("uniform_output_enabled", False)
    if uniform_output_enabled:
        result_path = os.path.join(cfg["Global"]["output_dir"], "train_result.json")
        if os.path.exists(result_path):
            try:
                os.remove(result_path)
            except OSError:
                pass
        cfg_dict = convert_to_dict(cfg)
        with open(os.path.join(cfg["Global"]["output_dir"], "config.yaml"), "w") as f:
            yaml.dump(cfg_dict, f)

    engine = Engine(cfg, mode="train")
    engine.train()
