import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
              "VGG11": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG11_pretrained.pdparams",
              "VGG13": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG13_pretrained.pdparams",
              "VGG16": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG16_pretrained.pdparams",
              "VGG19": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG19_pretrained.pdparams",
             }

__all__ = list(MODEL_URLS.keys())


class ConvBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, groups, name=None):
        super(ConvBlock, self).__init__()

        self.groups = groups
        self._conv_1 = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name=name + "1_weights"),
            bias_attr=False)
        if groups == 2 or groups == 3 or groups == 4:
            self._conv_2 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "2_weights"),
                bias_attr=False)
        if groups == 3 or groups == 4:
            self._conv_3 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "3_weights"),
                bias_attr=False)
        if groups == 4:
            self._conv_4 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "4_weights"),
                bias_attr=False)

        self._pool = MaxPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        if self.groups == 2 or self.groups == 3 or self.groups == 4:
            x = self._conv_2(x)
            x = F.relu(x)
        if self.groups == 3 or self.groups == 4:
            x = self._conv_3(x)
            x = F.relu(x)
        if self.groups == 4:
            x = self._conv_4(x)
            x = F.relu(x)
        x = self._pool(x)
        return x


class VGGNet(nn.Layer):
    def __init__(self, layers=11, stop_grad_layers=0, class_dim=1000):
        super(VGGNet, self).__init__()

        self.layers = layers
        self.stop_grad_layers = stop_grad_layers
        self.vgg_configure = {
            11: [1, 1, 2, 2, 2],
            13: [2, 2, 2, 2, 2],
            16: [2, 2, 3, 3, 3],
            19: [2, 2, 4, 4, 4]
        }
        assert self.layers in self.vgg_configure.keys(), \
            "supported layers are {} but input layer is {}".format(
                self.vgg_configure.keys(), layers)
        self.groups = self.vgg_configure[self.layers]

        self._conv_block_1 = ConvBlock(3, 64, self.groups[0], name="conv1_")
        self._conv_block_2 = ConvBlock(64, 128, self.groups[1], name="conv2_")
        self._conv_block_3 = ConvBlock(128, 256, self.groups[2], name="conv3_")
        self._conv_block_4 = ConvBlock(256, 512, self.groups[3], name="conv4_")
        self._conv_block_5 = ConvBlock(512, 512, self.groups[4], name="conv5_")

        for idx, block in enumerate([
                self._conv_block_1, self._conv_block_2, self._conv_block_3,
                self._conv_block_4, self._conv_block_5
        ]):
            if self.stop_grad_layers >= idx + 1:
                for param in block.parameters():
                    param.trainable = False

        self._drop = Dropout(p=0.5, mode="downscale_in_infer")
        self._fc1 = Linear(
            7 * 7 * 512,
            4096,
            weight_attr=ParamAttr(name="fc6_weights"),
            bias_attr=ParamAttr(name="fc6_offset"))
        self._fc2 = Linear(
            4096,
            4096,
            weight_attr=ParamAttr(name="fc7_weights"),
            bias_attr=ParamAttr(name="fc7_offset"))
        self._out = Linear(
            4096,
            class_dim,
            weight_attr=ParamAttr(name="fc8_weights"),
            bias_attr=ParamAttr(name="fc8_offset"))

    def forward(self, inputs):
        x = self._conv_block_1(inputs)
        x = self._conv_block_2(x)
        x = self._conv_block_3(x)
        x = self._conv_block_4(x)
        x = self._conv_block_5(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self._fc1(x)
        x = F.relu(x)
        x = self._drop(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._drop(x)
        x = self._out(x)
        return x

    
    
def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def VGG11(pretrained, model, model_url, use_ssld=False, **kwargs):
    model = VGGNet(layers=11, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["VGG11"], use_ssld=use_ssld)
    return model


def VGG13(pretrained, model, model_url, use_ssld=False, **kwargs):
    model = VGGNet(layers=13, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["VGG13"], use_ssld=use_ssld)
    return model


def VGG16(pretrained, model, model_url, use_ssld=False, **kwargs):
    model = VGGNet(layers=16, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["VGG16"], use_ssld=use_ssld)
    return model


def VGG19(pretrained, model, model_url, use_ssld=False, **kwargs):
    model = VGGNet(layers=19, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["VGG19"], use_ssld=use_ssld)
    return model
