from paddle.nn import Conv2D, Identity

from ..legendary_models.pp_lcnet_v2 import MODEL_URLS, PPLCNetV2_base, RepDepthwiseSeparable, _load_pretrained

__all__ = ["PPLCNetV2_base_ShiTu"]


def PPLCNetV2_base_ShiTu(pretrained=False, use_ssld=False, **kwargs):
    """
    An variant network of PPLCNetV2_base
    1. remove ReLU layer after last_conv
    2. add bias to last_conv
    3. change stride to 1 in last two RepDepthwiseSeparable Block
    """
    model = PPLCNetV2_base(pretrained=False, use_ssld=use_ssld, **kwargs)

    def remove_ReLU_function(conv, pattern):
        new_conv = Identity()
        return new_conv

    def add_bias_last_conv(conv, pattern):
        new_conv = Conv2D(
            in_channels=conv._in_channels,
            out_channels=conv._out_channels,
            kernel_size=conv._kernel_size,
            stride=conv._stride,
            padding=conv._padding,
            groups=conv._groups,
            bias_attr=True)
        return new_conv

    def last_stride_function(rep_block, pattern):
        new_conv = RepDepthwiseSeparable(
            in_channels=rep_block.in_channels,
            out_channels=rep_block.out_channels,
            stride=1,
            dw_size=rep_block.dw_size,
            split_pw=rep_block.split_pw,
            use_rep=rep_block.use_rep,
            use_se=rep_block.use_se,
            use_shortcut=rep_block.use_shortcut)
        return new_conv

    pattern_act = ["act"]
    pattern_lastconv = ["last_conv"]
    pattern_last_stride = [
        "stages[3][0]",
        "stages[3][1]",
    ]
    model.upgrade_sublayer(pattern_act, remove_ReLU_function)
    model.upgrade_sublayer(pattern_lastconv, add_bias_last_conv)
    model.upgrade_sublayer(pattern_last_stride, last_stride_function)

    # load params again after upgrade some layers
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNetV2_base"], use_ssld)
    return model
