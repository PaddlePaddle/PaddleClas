from abc import ABC
from paddle import nn
import re


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(TheseusLayer, self).__init__()
        self.res_dict = {}

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

    def update_res(self, return_patterns):
        if not return_patterns or isinstance(self, WrapLayer):
            return
        for layer_i in self._sub_layers:
            layer_name = self._sub_layers[layer_i].full_name()
            if isinstance(self._sub_layers[layer_i], (nn.Sequential, nn.LayerList)):
                self._sub_layers[layer_i] = wrap_theseus(self._sub_layers[layer_i], self.res_dict)
                self._sub_layers[layer_i].update_res(return_patterns)
            else:
                for return_pattern in return_patterns:
                    if re.match(return_pattern, layer_name):
                        if not isinstance(self._sub_layers[layer_i], TheseusLayer):
                            self._sub_layers[layer_i] = wrap_theseus(self._sub_layers[layer_i], self.res_dict)
                        else:
                            self._sub_layers[layer_i].res_dict = self.res_dict

                        self._sub_layers[layer_i].register_forward_post_hook(
                            self._sub_layers[layer_i]._save_sub_res_hook)
            if isinstance(self._sub_layers[layer_i], TheseusLayer):
                self._sub_layers[layer_i].res_dict = self.res_dict
                self._sub_layers[layer_i].update_res(return_patterns)

    def _save_sub_res_hook(self, layer, input, output):
        self.res_dict[layer.full_name()] = output

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"output": output}
        for res_key in list(self.res_dict):
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def replace_sub(self, layer_name_pattern, replace_function, recursive=True):
        for layer_i in self._sub_layers:
            layer_name = self._sub_layers[layer_i].full_name()
            if re.match(layer_name_pattern, layer_name):
                self._sub_layers[layer_i] = replace_function(self._sub_layers[layer_i])
            if recursive:
                if isinstance(self._sub_layers[layer_i], TheseusLayer):
                    self._sub_layers[layer_i].replace_sub(
                        layer_name_pattern, replace_function, recursive)
                elif isinstance(self._sub_layers[layer_i], (nn.Sequential, nn.LayerList)):
                    for layer_j in self._sub_layers[layer_i]._sub_layers:
                        self._sub_layers[layer_i]._sub_layers[layer_j].replace_sub(
                            layer_name_pattern, replace_function, recursive)

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


class WrapLayer(TheseusLayer):
    def __init__(self, sub_layer, res_dict=None):
        super(WrapLayer, self).__init__()
        self.sub_layer = sub_layer
        self.name = sub_layer.full_name()
        if res_dict is not None:
            self.res_dict = res_dict

    def full_name(self):
        return self.name

    def forward(self, *inputs, **kwargs):
        return self.sub_layer(*inputs, **kwargs)

    def update_res(self, return_patterns):
        if not return_patterns or not isinstance(self.sub_layer, (nn.Sequential, nn.LayerList)):
            return
        for layer_i in self.sub_layer._sub_layers:
            if isinstance(self.sub_layer._sub_layers[layer_i], (nn.Sequential, nn.LayerList)):
                self.sub_layer._sub_layers[layer_i] = wrap_theseus(self.sub_layer._sub_layers[layer_i], self.res_dict)
                self.sub_layer._sub_layers[layer_i].update_res(return_patterns)
            elif isinstance(self.sub_layer._sub_layers[layer_i], TheseusLayer):
                self.sub_layer._sub_layers[layer_i].res_dict = self.res_dict

            layer_name = self.sub_layer._sub_layers[layer_i].full_name()
            for return_pattern in return_patterns:
                if re.match(return_pattern, layer_name):
                    self.sub_layer._sub_layers[layer_i].register_forward_post_hook(
                        self._sub_layers[layer_i]._save_sub_res_hook)

            if isinstance(self.sub_layer._sub_layers[layer_i], TheseusLayer):
                self.sub_layer._sub_layers[layer_i].update_res(return_patterns)


def wrap_theseus(sub_layer, res_dict=None):
    wrapped_layer = WrapLayer(sub_layer, res_dict)
    return wrapped_layer
