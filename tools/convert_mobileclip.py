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
"""Convert official MobileCLIP weights and compare forward outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


def convert_name(name):
    replacements = {
        "image_encoder.model.": "image_encoder.",
        ".pos_embed.pos_embed": ".pos_embed",
        ".lkb_reparam.": ".conv.",
        ".reparam_conv.": ".conv.",
        ".pre_norm_mha.0.": ".norm1.",
        ".pre_norm_mha.1.": ".attn.",
        ".pre_norm_ffn.0.": ".norm2.",
        ".pre_norm_ffn.1.": ".mlp.fc1.",
        ".pre_norm_ffn.4.": ".mlp.fc2.",
        ".running_mean": "._mean",
        ".running_var": "._variance",
    }
    for source, target in replacements.items():
        name = name.replace(source, target)
    name = re.sub(r"\.se\.(?:fc1|reduce)\.", ".se.conv1.", name)
    return re.sub(r"\.se\.(?:fc2|expand)\.", ".se.conv2.", name)


def export_reference(args, output):
    import torch

    if not args.torch_repo or not args.checkpoint:
        raise ValueError("export requires --torch-repo and --checkpoint")
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(Path(args.torch_repo).resolve()))
    import mobileclip

    name = "mobileclip_b" if args.model == "mobileclip_blt" else args.model
    model, _, _ = mobileclip.create_model_and_transforms(
        name, pretrained=args.checkpoint, reparameterize=True, device="cpu")
    model.eval()
    state = {
        k: v.detach().cpu().numpy()
        for k, v in model.state_dict().items()
    }
    np.savez(output / "weights.npz", **state)
    digest = hashlib.sha256()
    with open(args.checkpoint, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "variant":
        args.model,
        "torch_version":
        torch.__version__,
        "reference_commit":
        subprocess.check_output(
            ["git", "-C", args.torch_repo, "rev-parse", "HEAD"],
            text=True).strip(),
        "checkpoint_sha256":
        digest.hexdigest(),
        "linear_weights": [
            name + ".weight" for name, layer in model.named_modules()
            if isinstance(layer, torch.nn.Linear)
        ],
        "seeds":
        args.seeds,
        "batch_size":
        2,
        "reference_reparameterized":
        True,
    }
    (output /
     "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    size = 224 if name == "mobileclip_b" else 256
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        image = rng.random((2, 3, size, size), dtype=np.float32)
        tokens = np.zeros((2, 77), dtype=np.int64)
        for row in range(2):
            end = int(rng.integers(8, 77))
            tokens[row, 0] = 49406
            tokens[row, 1:end] = rng.integers(1, 49406, size=end - 1)
            tokens[row, end] = 49407
        with torch.no_grad():
            ti, tt = torch.from_numpy(image), torch.from_numpy(tokens)
            raw_image, raw_text = model.encode_image(ti), model.encode_text(tt)
            image_features, text_features, scale = model(ti, tt)
            logits = scale * image_features @ text_features.T
        values = {
            "raw_image": raw_image,
            "raw_text": raw_text,
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": scale,
            "logits_per_image": logits,
            "logits_per_text": logits.T,
        }
        np.savez(
            output / "fixture_{}.npz".format(seed),
            image=image,
            tokens=tokens,
            **{k: v.cpu().numpy()
               for k, v in values.items()})


def convert_weights(args, output):
    import paddle
    from ppcls.arch.backbone.model_zoo.mobileclip import MobileCLIP

    paddle.set_device("cpu")
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata["variant"] != args.model:
        raise ValueError("Reference model does not match --model")
    model = MobileCLIP(args.model)
    linear = {convert_name(k) for k in metadata["linear_weights"]}
    expected_linear = {
        name + ".weight"
        for name, layer in model.named_sublayers()
        if isinstance(layer, paddle.nn.Linear)
    }
    if linear != expected_linear:
        raise ValueError("Linear layers do not match the reference model")
    state = {}
    with np.load(output / "weights.npz") as weights:
        for name in weights.files:
            if name.endswith(".num_batches_tracked"):
                continue
            target_name = convert_name(name)
            value = weights[name]
            state[target_name] = value.T.copy(
            ) if target_name in linear else value.copy()
    target = model.state_dict()
    if set(state) != set(target):
        raise ValueError("Missing: {}; unexpected: {}".format(
            sorted(set(target) - set(state)),
            sorted(set(state) - set(target))))
    for name, value in state.items():
        if list(value.shape) != list(target[name].shape):
            raise ValueError("Invalid parameter shape: {}".format(name))
    model.set_state_dict(state)
    path = output / (args.model + ".pdparams")
    paddle.save(model.state_dict(), str(path))
    model.set_state_dict(paddle.load(str(path)))
    image_state = {
        k: v
        for k, v in state.items() if k.startswith("image_encoder.")
    }
    paddle.save(image_state, str(output / (args.model + "_image.pdparams")))
    model.eval()
    report = {
        "variant": args.model,
        "paddle_version": paddle.__version__,
        "reference": metadata,
        "threshold": args.threshold,
        "device": "cpu",
        "strict_parameter_load": True,
        "results": [],
        "passed": True
    }
    if not metadata["seeds"]:
        raise ValueError("No reference inputs")
    for seed in metadata["seeds"]:
        with np.load(
                output /
                "fixture_{}.npz".format(seed)) as fixture, paddle.no_grad():
            image, tokens = paddle.to_tensor(
                fixture["image"]), paddle.to_tensor(fixture["tokens"])
            raw_image, raw_text = model.encode_image(image), model.encode_text(
                tokens)
            image_features, text_features, scale = model(image, tokens)
            logits = scale * image_features @ text_features.transpose([1, 0])
            values = {
                "raw_image": raw_image,
                "raw_text": raw_text,
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": scale,
                "logits_per_image": logits,
                "logits_per_text": logits.transpose([1, 0])
            }
            result = {"seed": seed, "metrics": {}}
            for name, value in values.items():
                actual, expected = value.numpy(), fixture[name]
                if actual.shape != expected.shape or not np.isfinite(
                        actual).all():
                    raise ValueError("Invalid output: {}".format(name))
                diff = np.abs(
                    actual.astype(np.float64) - expected.astype(np.float64))
                passed = float(diff.max()) < args.threshold
                result["metrics"][name] = {
                    "max_abs_diff": float(diff.max()),
                    "passed": passed
                }
                report["passed"] &= passed
            report["results"].append(result)
    (output / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=["export", "convert"])
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b",
            "mobileclip_blt"
        ])
    parser.add_argument("--torch-repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 42, 7])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=1e-4)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    output = Path(args.output_dir) / args.model
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "export":
        export_reference(args, output)
    else:
        convert_weights(args, output)
