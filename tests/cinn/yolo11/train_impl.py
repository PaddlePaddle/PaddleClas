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

"""Full-parameter fine-tuning pilot on a fixed real ImageNet subset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["dynamic", "cinn"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--data", type=Path)
p.add_argument("--resume", type=Path)
args = p.parse_args()
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["FLAGS_cudnn_deterministic"] = "true"
os.environ["FLAGS_cudnn_exhaustive_search"] = "false"
os.environ["FLAGS_use_cinn"] = str(args.mode == "cinn").lower()
os.environ["OMP_NUM_THREADS"] = "1"
root = Path(__file__).parent
sys.path.insert(0, os.environ["PADDLECLAS_DIR"])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from checkpoint_utils import strict_set_state_dict
import numpy as np
import paddle
from paddle.base import core
from ppcls.arch.backbone.model_zoo.yolo11 import YOLO11_cls_n

if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
args.output.mkdir(parents=True)
paddle.set_device("gpu:0")
paddle.seed(20260915)
core._set_prim_forward_blacklist("pd_op.batch_norm", "pd_op.batch_norm_")
data_path = args.data or Path(os.environ["DATA_DIR"]) / "data.npz"
with np.load(data_path) as data:
    tx, ty, vx, vy = [data[k] for k in ["train_x", "train_y", "val_x", "val_y"]]
checkpoint = Path(os.environ["TRAIN_CHECKPOINT"])
model = YOLO11_cls_n(pretrained=False)
state = paddle.load(str(checkpoint))
strict_set_state_dict(model, state)
optimizer = paddle.optimizer.AdamW(
    learning_rate=1e-5, weight_decay=0.01, parameters=model.parameters()
)
batch = 16
if not len(tx) or not len(vx) or len(tx) % batch or len(vx) % batch:
    raise ValueError(
        f"Train/validation sample counts must be positive multiples of batch size {batch}"
    )
if args.mode == "cinn":
    model = paddle.jit.to_static(
        model,
        full_graph=True,
        backend="CINN",
        input_spec=[paddle.static.InputSpec([batch, 3, 224, 224], "float32")],
    )
config = dict(
    mode=args.mode,
    epochs=args.epochs,
    batch=batch,
    seed=20260915,
    optimizer="AdamW",
    lr=1e-5,
    weight_decay=0.01,
    precision="FP32; NVIDIA_TF32_OVERRIDE=0",
    native_bn=True,
    train_images=len(tx),
    validation_images=len(vx),
    paddle=paddle.__version__,
    paddle_commit=paddle.version.commit,
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    manifest_sha256=hashlib.sha256(
        (data_path.parent / "manifest.json").read_bytes()
    ).hexdigest(),
    scope="ImageNet100 subset full-parameter fine-tuning pilot; not full ImageNet convergence",
)
(args.output / "config.json").write_text(json.dumps(config, indent=2))
import random

start_epoch = 0
if args.resume:
    saved = paddle.load(str(args.resume))
    for key in [
        "mode",
        "batch",
        "seed",
        "lr",
        "weight_decay",
        "native_bn",
        "paddle_commit",
        "checkpoint_sha256",
        "manifest_sha256",
    ]:
        if saved["config"].get(key) != config.get(key):
            raise ValueError(f"Resume mismatch: {key}")
    strict_set_state_dict(model, saved["model"])
    optimizer.set_state_dict(saved["optimizer"])
    start_epoch = saved["epoch"]
    if args.epochs <= start_epoch:
        raise ValueError("Target epochs must exceed saved epoch")
    paddle.set_rng_state(saved["cpu_rng"])
    paddle.set_cuda_rng_state(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
config.update(start_epoch=start_epoch, resume=str(args.resume) if args.resume else None)
(args.output / "config.json").write_text(json.dumps(config, indent=2))


def save_training_state(epoch, step):
    temporary = args.output / "training_state.tmp"
    paddle.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            step=step,
            config=config,
            cpu_rng=paddle.get_rng_state(),
            cuda_rng=paddle.get_cuda_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        ),
        str(temporary),
    )
    os.replace(temporary, args.output / "training_state.pdstate")


def tensors(images, labels):
    x = np.ascontiguousarray(images.transpose(0, 3, 1, 2), dtype="float32") / 255.0
    return paddle.to_tensor(x), paddle.to_tensor(labels)


def evaluate(epoch):
    model.eval()
    losses = []
    correct = 0
    start = time.perf_counter()
    with paddle.no_grad():
        for offset in range(0, len(vx), batch):
            x, y = tensors(vx[offset : offset + batch], vy[offset : offset + batch])
            logits = model(x)
            loss = paddle.nn.functional.cross_entropy(logits, y)
            losses.append(float(loss.item()))
            correct += int((logits.argmax(-1) == y).astype("int64").sum().item())
            if offset == 0:
                np.save(args.output / f"eval_epoch_{epoch:02d}.npy", logits.numpy())
    row = dict(
        epoch=epoch,
        val_loss=float(np.mean(losses)),
        top1=correct / len(vy),
        seconds=time.perf_counter() - start,
    )
    if not np.isfinite(row["val_loss"]):
        raise RuntimeError(row)
    print(json.dumps(row), flush=True)
    return row


evaluation = []
evaluation.append(evaluate(start_epoch))
(args.output / "eval.json").write_text(json.dumps(evaluation, indent=2))
step = start_epoch * (len(tx) // batch)
with (args.output / "loss.jsonl").open("w") as log:
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        indices = np.random.default_rng(20260915 + epoch).permutation(len(tx))
        for offset in range(0, len(tx), batch):
            selected = indices[offset : offset + batch]
            x, y = tensors(tx[selected], ty[selected])
            paddle.device.cuda.synchronize()
            start = time.perf_counter()
            logits = model(x)
            loss = paddle.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            paddle.device.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            value = float(loss.item())
            if not np.isfinite(value):
                raise RuntimeError(f"Nonfinite loss at {step}")
            row = dict(step=step, epoch=epoch, loss=value, train_ms=elapsed)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % 50 == 0:
                print(json.dumps(row), flush=True)
            step += 1
        evaluation.append(evaluate(epoch))
        (args.output / "eval.json").write_text(json.dumps(evaluation, indent=2))
        save_training_state(epoch, step)
paddle.save(model.state_dict(), str(args.output / "final.pdparams"))
print("FINISHED", step, flush=True)
