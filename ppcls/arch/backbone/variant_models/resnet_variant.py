from paddle.nn import Conv2D
from ppcls.arch.backbone.legendary_models.resnet import ResNet50, MODEL_URLS, _load_pretrained

__all__ = ["ResNet50_last_stage_stride1"]


def ResNet50_last_stage_stride1(pretrained=False, use_ssld=False, **kwargs):
    def replace_function(conv, pattern):
        new_conv = Conv2D(
            in_channels=conv._in_channels,
            out_channels=conv._out_channels,
            kernel_size=conv._kernel_size,
            stride=1,
            padding=conv._padding,
            groups=conv._groups,
            bias_attr=conv._bias_attr)
        return new_conv

    pattern = ["blocks[13].conv1.conv", "blocks[13].short.conv"]
    model = ResNet50(pretrained=False, use_ssld=use_ssld, **kwargs)
    model.upgrade_sublayer(pattern, replace_function)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet50"], use_ssld)
    return model
