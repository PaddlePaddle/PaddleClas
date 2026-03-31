import argparse

import numpy as np


def main(output, seed=2026, input_size=224):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 3, input_size, input_size), dtype=np.float32)
    np.savez(output, x=x)
    print(f"saved: {output}")
    print(f"x shape: {tuple(x.shape)}")
    print(f"x mean: {float(x.mean()):.8f}")
    print(f"x std: {float(x.std()):.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input_size", type=int, default=224)
    args = parser.parse_args()
    main(args.output, args.seed, args.input_size)
