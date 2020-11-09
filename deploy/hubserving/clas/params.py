# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config(object):
    pass


def read_params():
    cfg = Config()

    cfg.model_file = "./inference/cls_infer.pdmodel"
    cfg.params_file = "./inference/cls_infer.pdiparams"
    cfg.batch_size = 1
    cfg.use_gpu = False
    cfg.ir_optim = True
    cfg.gpu_mem = 8000
    cfg.use_fp16 = False
    cfg.use_tensorrt = False

    # params for preprocess
    cfg.resize_short = 256
    cfg.resize = 224
    cfg.normalize = True

    return cfg
