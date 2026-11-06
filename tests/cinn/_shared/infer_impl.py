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

"""Deterministic inference on a fixed official-validation batch."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dynamic", "cinn"], default="dynamic")
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument(
    "--model", choices=["ghostnetv3", "naflexvit", "mobilenetv5"], required=True
)
p.add_argument("--native-reductions", action="store_true")
p.add_argument("--native-linear", action="store_true")
p.add_argument("--native-pool", action="store_true")
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["FLAGS_cudnn_deterministic"] = "true"
sys.path.insert(0, os.environ["PADDLECLAS_DIR"])
import numpy as np
import paddle
from paddle.base import core
from ppcls.arch.backbone.model_zoo.ghostnet_v3 import GhostNetV3_x1_0
from ppcls.arch.backbone.model_zoo.naflexvit import naflexvit_base_patch16_gap
from ppcls.arch.backbone.legendary_models.mobilenet_v5 import MobileNetV5_base
from ppcls.utils import logger

logger.init_logger()

paddle.set_device("gpu:0")
core._set_prim_forward_blacklist("pd_op.batch_norm", "pd_op.batch_norm_")
if args.native_reductions:
    core._set_prim_forward_blacklist("pd_op.layer_norm", "pd_op.softmax")
if args.native_linear:
    core._set_prim_forward_blacklist("pd_op.linear_v2")
if args.native_pool:
    core._set_prim_forward_blacklist("pd_op.mean")
model = {
    "ghostnetv3": GhostNetV3_x1_0,
    "naflexvit": naflexvit_base_patch16_gap,
    "mobilenetv5": MobileNetV5_base,
}[args.model](pretrained=False, drop_rate=0.0)
state = paddle.load(str(args.checkpoint))
assert state.keys() == model.state_dict().keys()
model.set_state_dict(state)
model.eval()
if args.mode == "cinn":
    model = paddle.jit.to_static(model, full_graph=True, backend="CINN")
with np.load(Path(os.environ["DATA_DIR"]) / "data.npz") as data:
    x = (
        np.ascontiguousarray(data["val_x"][:16].transpose(0, 3, 1, 2), dtype="float32")
        / 255.0
    )
    mean, std = (
        ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if args.model == "ghostnetv3"
        else (
            ([0.5] * 3, [0.5] * 3)
            if args.model == "naflexvit"
            else ([0.0] * 3, [1.0] * 3)
        )
    )
    x = paddle.to_tensor(
        (x - np.array(mean, dtype="float32")[None, :, None, None])
        / np.array(std, dtype="float32")[None, :, None, None]
    )
    targets = data["val_y"][:16]
with paddle.no_grad():
    start = time.perf_counter()
    logits = model(x)
    paddle.device.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1000
    for _ in range(5):
        logits = model(x)
    paddle.device.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        logits = model(x)
    paddle.device.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000 / 50
predictions = logits.argmax(-1).numpy()
print(
    json.dumps(
        dict(
            mode=args.mode,
            checkpoint=str(args.checkpoint),
            first_call_ms=first_ms,
            native_reductions=args.native_reductions,
            native_linear=args.native_linear,
            native_pool=args.native_pool,
            steady_batch_ms=ms,
            images_per_second=16 * 1000 / ms,
            predictions=predictions.tolist(),
            targets=targets.tolist(),
            batch_top1=float(np.mean(predictions == targets)),
        )
    )
)

np.save(args.output / "logits.npy", logits.numpy())
(args.output / "inference.json").write_text(
    json.dumps(
        dict(
            steady_batch_ms=ms,
            images_per_second=16 * 1000 / ms,
            first_call_ms=first_ms,
            batch=16,
            warmup=5,
            iterations=50,
        )
    )
)
