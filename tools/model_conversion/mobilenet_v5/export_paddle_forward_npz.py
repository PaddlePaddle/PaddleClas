import argparse

import numpy as np
import paddle

from ppcls.arch.backbone import MobileNetV5_300M, MobileNetV5_300M_enc, MobileNetV5_base
from ppcls.utils import logger


def main(variant, pdparams, ref_npz, output):
    logger.init_logger()

    if variant == "mobilenetv5_300m":
        model = MobileNetV5_300M(class_num=0)
    elif variant == "mobilenetv5_300m_enc":
        model = MobileNetV5_300M_enc()
    elif variant == "mobilenetv5_base":
        model = MobileNetV5_base(class_num=1000)
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    state = paddle.load(pdparams)
    model.set_state_dict(state)
    model.eval()

    ref = np.load(ref_npz)
    x = ref["x"].astype("float32")

    with paddle.no_grad():
        x_tensor = paddle.to_tensor(x)
        feat = model.forward_features(x_tensor).numpy()
        out = model(x_tensor).numpy()

    np.savez(output, x=x, feat=feat, out=out, y=out)
    print(f"saved: {output}")
    print(f"x shape: {tuple(x.shape)}")
    print(f"feat shape: {tuple(feat.shape)}")
    print(f"out shape: {tuple(out.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="mobilenetv5_300m")
    parser.add_argument("--pdparams", type=str, required=True)
    parser.add_argument("--ref_npz", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    main(args.variant, args.pdparams, args.ref_npz, args.output)
