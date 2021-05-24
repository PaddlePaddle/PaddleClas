from abc import ABC
from paddle import nn
import re


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer, ABC):
    def __init__(self, *args, return_patterns=None, stop_layer=None, **kwargs):
        super(TheseusLayer, self).__init__()
        self.res_dict = None
        self.register_forward_post_hook(self._disconnect_res_dict_hook)
        if return_patterns is not None or stop_layer is not None:
            self._update_sub(return_patterns, stop_layer)

    def forward(self, *input, res_dict=None, **kwargs):
        if res_dict is not None:
            self.res_dict = res_dict

    def _update_sub(self, return_layers, stop_layer):
        after_stop = False
        for layer_i in self._sub_layers:
            layer_name = self._sub_layers[layer_i].full_name()
            if stop_layer is not None and layer_name == stop_layer:
                after_stop = True
            if after_stop:
                self._sub_layers[layer_i] = Identity()
            for return_pattern in return_layers:
                if return_layers is not None and re.match(return_pattern, layer_name):
                    self._sub_layers[layer_i].register_forward_post_hook(self._save_sub_res_hook)

    def _save_sub_res_hook(self, layer, input, output):
        self.res_dict[layer.full_name()] = output

    def _disconnect_res_dict_hook(self, input, output):
        self.res_dict = None

    def replace_sub(self, layer_name_pattern, replace_function, recursive=True):
        for layer_i in self._sub_layers:
            layer_name = self._sub_layers[layer_i].full_name()
            if re.match(layer_name_pattern, layer_name):
                self._sub_layers[layer_i] = replace_function(self._sub_layers[layer_i])
            if recursive and isinstance(self._sub_layers[layer_i], TheseusLayer):
                self._sub_layers[layer_i].replace_sub(layer_name_pattern, replace_function, recursive)

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