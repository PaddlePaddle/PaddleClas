from abc import ABC
from paddle import nn
import re


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer):
    def __init__(self, *args, return_patterns=None, **kwargs):
        super(TheseusLayer, self).__init__()
        self.res_dict = None
        if return_patterns is not None:
            self._update_res(return_patterns)

    def forward(self, *input, res_dict=None, **kwargs):
        if res_dict is not None:
            self.res_dict = res_dict

    # stop doesn't work when stop layer has a parallel branch.
    def stop_after(self, stop_layer_name: str):
        after_stop = False
        for layer_i in self._sub_layers:
            if after_stop:
                self._sub_layers[layer_i] = Identity()
                continue
            layer_name = self._sub_layers[layer_i].full_name()
            if layer_name == stop_layer_name:
                after_stop = True
                continue
            if isinstance(self._sub_layers[layer_i], TheseusLayer):
                after_stop = self._sub_layers[layer_i].stop_after(
                    stop_layer_name)
        return after_stop

    def _update_res(self, return_layers):
        for layer_i in self._sub_layers:
            layer_name = self._sub_layers[layer_i].full_name()
            for return_pattern in return_layers:
                if return_layers is not None and re.match(return_pattern,
                                                          layer_name):
                    self._sub_layers[layer_i].register_forward_post_hook(
                        self._save_sub_res_hook)

    def replace_sub(self, layer_name_pattern, replace_function,
                    recursive=True):
        for k in self._sub_layers.keys():
            layer_name = self._sub_layers[k].full_name()
            if re.match(layer_name_pattern, layer_name):
                self._sub_layers[k] = replace_function(self._sub_layers[k])
            if recursive:
                if isinstance(self._sub_layers[k], TheseusLayer):
                    self._sub_layers[k].replace_sub(
                        layer_name_pattern, replace_function, recursive)
                elif isinstance(self._sub_layers[k],
                                nn.Sequential) or isinstance(
                                    self._sub_layers[k], nn.LayerList):
                    for kk in self._sub_layers[k]._sub_layers.keys():
                        self._sub_layers[k]._sub_layers[kk].replace_sub(
                            layer_name_pattern, replace_function, recursive)
                else:
                    pass

    '''
    example of replace function:
    def replace_conv(origin_conv: nn.Conv2D):
        new_conv = nn.Conv2D(
            in_channels=origin_conv._in_channels,
            out_channels=origin_conv._out_channels,
            kernel_size=origin_conv._kernel_size,
            stride=2
        )
        return new_conv

        '''
