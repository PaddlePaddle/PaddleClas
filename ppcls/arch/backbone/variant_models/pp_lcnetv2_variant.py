from paddle.nn import Conv2D, Identity
from ..legendary_models.pp_lcnet_v2 import PPLCNetV2_base, RepDepthwiseSeparable, MODEL_URLS, _load_pretrained

__all__ = ["PPLCNetV2_base_ShiTu"]


def PPLCNetV2_base_ShiTu(pretrained=False, use_ssld=False, **kwargs):

    model = PPLCNetV2_base(pretrained=False, use_ssld=use_ssld, **kwargs)

    def remove_ReLU_function(conv, pattern):
        new_conv = Identity()
        return new_conv

    # def last_stride_function(conv, pattern):
    #     new_conv = Conv2D(
    #         weight_attr=conv._param_attr,
    #         in_channels=conv._in_channels,
    #         out_channels=conv._out_channels,
    #         kernel_size=conv._kernel_size,
    #         stride=1,
    #         padding=conv._padding,
    #         groups=conv._groups,
    #         bias_attr=conv._bias_attr)
    #     return new_conv

    pattern_act = ["act"]
    # pattern_last_stride = [
    #     "stages[3][0].dw_conv_list[0].conv",
    #     "stages[3][0].dw_conv_list[1].conv",
    #     "stages[3][0].dw_conv",
    #     "stages[3][0].pw_conv.conv",
    #     "stages[3][1].dw_conv_list[0].conv",
    #     "stages[3][1].dw_conv_list[1].conv",
    #     "stages[3][1].dw_conv_list[2].conv",
    #     "stages[3][1].dw_conv",
    #     "stages[3][1].pw_conv.conv",
    # ]
    # model.upgrade_sublayer(pattern_last_stride, last_stride_function) # TODO: theseuslayer有BUG，暂时注释掉
    model.upgrade_sublayer(pattern_act, remove_ReLU_function)

    # load params again after upgrade some layers
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNetV2_base"], use_ssld)
    return model
