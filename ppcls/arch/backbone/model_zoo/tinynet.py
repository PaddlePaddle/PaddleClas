# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was based on https://gitee.com/mindspore/models/tree/master/research/cv/tinynet
# reference: https://arxiv.org/abs/2010.14819

import paddle.nn as nn

from .efficientnet import EfficientNet, efficientnet
from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "TinyNet_A":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TinyNet_A_pretrained.pdparams",
    "TinyNet_B":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TinyNet_B_pretrained.pdparams",
    "TinyNet_C":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TinyNet_C_pretrained.pdparams",
    "TinyNet_D":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TinyNet_D_pretrained.pdparams",
    "TinyNet_E":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TinyNet_E_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


def tinynet_params(model_name):
    """ Map TinyNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients: width,depth,resolution,dropout
        "tinynet-a": (1.00, 1.200, 192, 0.2),
        "tinynet-b": (0.75, 1.100, 188, 0.2),
        "tinynet-c": (0.54, 0.850, 184, 0.2),
        "tinynet-d": (0.54, 0.695, 152, 0.2),
        "tinynet-e": (0.51, 0.600, 106, 0.2),
    }
    return params_dict[model_name]


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('tinynet'):
        w, d, _, p = tinynet_params(model_name)
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' %
                                  model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


class TinyNet(EfficientNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            fin_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
            std = (2 / fin_in)**0.5
            nn.initializer.Normal(std=std)(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.Linear):
            fin_in = m.weight.shape[0]
            bound = 1 / fin_in**0.5
            nn.initializer.Uniform(-bound, bound)(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def TinyNet_A(padding_type='DYNAMIC',
              override_params=None,
              use_se=True,
              pretrained=False,
              use_ssld=False,
              **kwargs):
    block_args, global_params = get_model_params("tinynet-a", override_params)
    model = TinyNet(
        block_args,
        global_params,
        name='a',
        padding_type=padding_type,
        use_se=use_se,
        fix_stem=True,
        num_features=1280,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["TinyNet_A"], use_ssld)
    return model


def TinyNet_B(padding_type='DYNAMIC',
              override_params=None,
              use_se=True,
              pretrained=False,
              use_ssld=False,
              **kwargs):
    block_args, global_params = get_model_params("tinynet-b", override_params)
    model = TinyNet(
        block_args,
        global_params,
        name='b',
        padding_type=padding_type,
        use_se=use_se,
        fix_stem=True,
        num_features=1280,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["TinyNet_B"], use_ssld)
    return model


def TinyNet_C(padding_type='DYNAMIC',
              override_params=None,
              use_se=True,
              pretrained=False,
              use_ssld=False,
              **kwargs):
    block_args, global_params = get_model_params("tinynet-c", override_params)
    model = TinyNet(
        block_args,
        global_params,
        name='c',
        padding_type=padding_type,
        use_se=use_se,
        fix_stem=True,
        num_features=1280,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["TinyNet_C"], use_ssld)
    return model


def TinyNet_D(padding_type='DYNAMIC',
              override_params=None,
              use_se=True,
              pretrained=False,
              use_ssld=False,
              **kwargs):
    block_args, global_params = get_model_params("tinynet-d", override_params)
    model = TinyNet(
        block_args,
        global_params,
        name='d',
        padding_type=padding_type,
        use_se=use_se,
        fix_stem=True,
        num_features=1280,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["TinyNet_D"], use_ssld)
    return model


def TinyNet_E(padding_type='DYNAMIC',
              override_params=None,
              use_se=True,
              pretrained=False,
              use_ssld=False,
              **kwargs):
    block_args, global_params = get_model_params("tinynet-e", override_params)
    model = TinyNet(
        block_args,
        global_params,
        name='e',
        padding_type=padding_type,
        use_se=use_se,
        fix_stem=True,
        num_features=1280,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["TinyNet_E"], use_ssld)
    return model
