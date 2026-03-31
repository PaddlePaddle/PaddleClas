import argparse

import numpy as np
import timm
import torch


def _load_npz_state(model, npz_path):
    data = np.load(npz_path, allow_pickle=False)
    sd = model.state_dict()
    loaded = 0
    for k in sd.keys():
        if k in data.files:
            sd[k] = torch.from_numpy(data[k])
            loaded += 1
    model.load_state_dict(sd, strict=False)
    print(f"loaded from npz: {npz_path}, keys={loaded}")


def _load_input(input_npz, seed, input_size):
    if input_npz is not None:
        data = np.load(input_npz, allow_pickle=False)
        return data["x"].astype("float32")
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.randn(1, 3, input_size, input_size).cpu().numpy()


def main(variant,
         output,
         seed=2026,
         input_size=256,
         state_npz=None,
         input_npz=None):
    model = timm.create_model(variant, pretrained=False)
    if state_npz is not None:
        _load_npz_state(model, state_npz)
    model.eval()

    x_np = _load_input(input_npz, seed, input_size)
    x = torch.from_numpy(x_np)
    with torch.no_grad():
        feat = model.forward_features(x)
        out = model(x)

    np.savez(
        output,
        x=x_np,
        feat=feat.cpu().numpy(),
        out=out.cpu().numpy(),
        y=out.cpu().numpy())
    print(f"saved: {output}")
    print(f"x shape: {tuple(x.shape)}")
    print(f"feat shape: {tuple(feat.shape)}")
    print(f"out shape: {tuple(out.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="mobilenetv5_300m")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--state_npz", type=str, default=None)
    parser.add_argument("--input_npz", type=str, default=None)
    args = parser.parse_args()

    main(args.variant, args.output, args.seed, args.input_size, args.state_npz,
         args.input_npz)
