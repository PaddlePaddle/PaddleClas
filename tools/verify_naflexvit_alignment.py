#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


TORCH_HELPER = r"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import timm  # noqa: E402

variant = sys.argv[2]
input_path = sys.argv[3]
state_path = sys.argv[4]
output_path = sys.argv[5]
device = sys.argv[6]

torch.manual_seed(2026)
torch.set_grad_enabled(False)
model = timm.create_model(variant, pretrained=False).to(device)
model.eval()

x = torch.from_numpy(np.load(input_path)).float().to(device)
state_dict = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
np.savez(state_path, **state_dict)

features = model.forward_features(x).detach().cpu().numpy()
logits = model.forward_head(torch.from_numpy(features).to(device)).detach().cpu().numpy()
np.savez(output_path, features=features, logits=logits)

print(json.dumps({"num_params": len(state_dict), "feature_shape": list(features.shape), "logit_shape": list(logits.shape)}))
"""


PADDLE_HELPER = r"""
import json
import sys
from pathlib import Path

import numpy as np
import paddle

repo_root = Path(sys.argv[1])
variant = sys.argv[2]
input_path = sys.argv[3]
state_path = sys.argv[4]
output_path = sys.argv[5]
device = sys.argv[6]

sys.path.insert(0, str(repo_root))

from ppcls.arch.backbone import __dict__ as backbone_dict  # noqa: E402

paddle.seed(2026)
paddle.set_device(device)
model = backbone_dict[variant](pretrained=False)
model.eval()

torch_state = np.load(state_path)
target_state = model.state_dict()
linear_weight_keys = {
    f"{name}.weight"
    for name, layer in model.named_sublayers()
    if isinstance(layer, paddle.nn.Linear)
}
converted = {}
missing = []
for key, value in target_state.items():
    if key not in torch_state:
        missing.append(key)
        continue
    array = torch_state[key]
    if key in linear_weight_keys:
        array = array.T
    if list(array.shape) != list(value.shape):
        raise ValueError(f"Shape mismatch for {key}: torch={array.shape}, paddle={value.shape}")
    converted[key] = paddle.to_tensor(array, place=value.place)

if missing:
    raise ValueError("Missing keys: " + ", ".join(missing))

model.set_state_dict(converted)

x = paddle.to_tensor(np.load(input_path), dtype="float32")
features = model.forward_features(x).numpy()
logits = model.forward_head(paddle.to_tensor(features, dtype="float32")).numpy()
np.savez(output_path, features=features, logits=logits)

print(json.dumps({"num_params": len(converted), "feature_shape": list(features.shape), "logit_shape": list(logits.shape)}))
"""


def run_helper(python_bin, helper_code, args):
    cmd = [python_bin, "-c", helper_code] + [str(x) for x in args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Helper failed:\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                result.stdout, result.stderr
            )
        )
    stdout = result.stdout.strip().splitlines()
    info = json.loads(stdout[-1]) if stdout else {}
    return info


def diff_metrics(a, b):
    diff = np.abs(a - b)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Verify NaFlexViT forward alignment between timm and PaddleClas.")
    parser.add_argument("--variant", default="naflexvit_base_patch16_gap", choices=[
        "naflexvit_base_patch16_gap",
        "naflexvit_base_patch16_par_gap",
        "naflexvit_base_patch16_parfac_gap",
    ])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timm-root", default=str((Path(__file__).resolve().parents[2] / "pytorch-image-models")))
    parser.add_argument("--torch-python", default=str((Path(__file__).resolve().parents[2] / "timm_env" / "bin" / "python")))
    parser.add_argument("--paddle-python", default=str((Path(__file__).resolve().parents[2] / "paddleclas_env" / "bin" / "python")))
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--paddle-device", default="gpu" if os.path.exists("/dev/nvidia0") else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    x = np.random.RandomState(2026).standard_normal(
        (args.batch_size, 3, args.height, args.width)
    ).astype("float32")

    with tempfile.TemporaryDirectory(prefix="naflexvit_align_") as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.npy"
        state_path = tmpdir / "torch_state.npz"
        torch_output_path = tmpdir / "torch_output.npz"
        paddle_output_path = tmpdir / "paddle_output.npz"
        np.save(input_path, x)

        torch_info = run_helper(
            args.torch_python,
            TORCH_HELPER,
            [args.timm_root, args.variant, input_path, state_path, torch_output_path, args.torch_device],
        )
        paddle_info = run_helper(
            args.paddle_python,
            PADDLE_HELPER,
            [repo_root, args.variant, input_path, state_path, paddle_output_path, args.paddle_device],
        )

        torch_output = np.load(torch_output_path)
        paddle_output = np.load(paddle_output_path)
        feature_metrics = diff_metrics(torch_output["features"], paddle_output["features"])
        logit_metrics = diff_metrics(torch_output["logits"], paddle_output["logits"])

    result = {
        "variant": args.variant,
        "input_shape": [args.batch_size, 3, args.height, args.width],
        "torch": torch_info,
        "paddle": paddle_info,
        "devices": {
            "torch": args.torch_device,
            "paddle": args.paddle_device,
        },
        "features": feature_metrics,
        "logits": logit_metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
