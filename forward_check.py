import torch
import paddle
import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
from paddle import nn
from ppcls.arch.backbone.legendary_models import open_clip as clip_paddle
import open_clip as clip_torch

MODEL_param = {
    "react": "pretrain_model_torch/react.pt",
    "laclip": "pretrain_model_torch/laclip.pt",
    "unicom": "pretrain_model_torch/unicom.pt",
}


def torch_to_numpy(name, x: torch.nn.Module):
    try:
        ckpt = torch.load(MODEL_param[name])["state_dict"]
    except:
        ckpt = torch.load(MODEL_param[name])
    x.load_state_dict(ckpt)
    state_dict_vision = x.visual.state_dict()
    state_numpy = {}
    for k in state_dict_vision.keys():
        state_numpy[k] = state_dict_vision[k].numpy()
    return x.visual, state_numpy


def numpy_to_paddle(state_dict_numpy, x: paddle.nn.Layer):
    state_paddle = {}
    a = x.visual.state_dict()
    for k in state_dict_numpy.keys():
        state_paddle[k] = paddle.to_tensor(state_dict_numpy[k])

    x.visual.set_state_dict(state_paddle)
    return x.visual


if __name__ == "__main__":
    torch.manual_seed(0)
    paddle.seed(0)
    vit_b_32_paddle = clip_paddle.Laclip_vit_b_32()
    vit_b_32_torch = clip_torch.create_model("ViT-B-32")
    vision_torch, state_dict_paddle = torch_to_numpy("laclip", vit_b_32_torch)
    vision_paddle = numpy_to_paddle(state_dict_paddle, vit_b_32_paddle)
    input = torch.ones([1, 3, 224, 224]).numpy()
    reprod_logger = ReprodLogger()
    vision_torch.eval()
    vision_paddle.eval()

    paddle_out = vision_paddle(paddle.to_tensor(input))
    reprod_logger.add("logits", paddle_out.numpy())
    reprod_logger.save("./result/forward_paddle.npy")

    torch_out = vision_torch(torch.Tensor(input))
    reprod_logger.add("logits", torch_out.detach().numpy())
    reprod_logger.save("./result/forward_torch.npy")

    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/forward_torch.npy")
    paddle_info = diff_helper.load_info("./result/forward_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="./result/log/forward_diff.log", diff_threshold=1e-5)
