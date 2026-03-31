import argparse

import numpy as np


def _compare_key(torch_data, paddle_data, key):
    if key not in torch_data or key not in paddle_data:
        print(f"{key}: skipped")
        return

    t = torch_data[key]
    p = paddle_data[key]
    if t.shape != p.shape:
        raise ValueError(
            f"{key} shape mismatch: torch {t.shape}, paddle {p.shape}")

    diff = np.abs(t - p)
    rel = diff / np.maximum(np.abs(t), 1e-8)
    print(f"[{key}]")
    print(f"shape: {t.shape}")
    print(f"max_abs_diff: {diff.max():.8f}")
    print(f"mean_abs_diff: {diff.mean():.8f}")
    print(f"p99_abs_diff: {np.percentile(diff, 99):.8f}")
    print(f"max_rel_diff: {rel.max():.8f}")
    print(f"mean_rel_diff: {rel.mean():.8f}")


def main(torch_npz, paddle_npz):
    torch_data = np.load(torch_npz, allow_pickle=False)
    paddle_data = np.load(paddle_npz, allow_pickle=False)
    for key in ["feat", "out", "y"]:
        _compare_key(torch_data, paddle_data, key)
        if key in torch_data and key in paddle_data:
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_npz", type=str, required=True)
    parser.add_argument("--paddle_npz", type=str, required=True)
    args = parser.parse_args()
    main(args.torch_npz, args.paddle_npz)
