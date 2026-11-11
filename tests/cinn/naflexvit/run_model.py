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

import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["train", "infer"], required=True)
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--epochs", default="1")
a = p.parse_args()
root = Path(__file__).parent
cmd = [
    sys.executable,
    str(root.parent / "_shared" / (a.task + "_impl.py")),
    "--mode",
    a.mode,
    "--output",
    a.output,
]
cmd += ["--model", "naflexvit"]
if a.task == "train":
    os.environ["TRAIN_CHECKPOINT"] = a.checkpoint
    cmd += [
        "--epochs",
        a.epochs,
        "--data",
        str(Path(os.environ["DATA_DIR"]) / "data.npz"),
    ]
    if os.environ.get("RESUME"):
        cmd += ["--resume", os.environ["RESUME"]]
else:
    cmd += ["--checkpoint", a.checkpoint]
subprocess.run(cmd, check=True)
