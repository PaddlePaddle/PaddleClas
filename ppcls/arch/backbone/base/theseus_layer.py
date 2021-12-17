from typing import List, Union, Callable, Any
from paddle import nn
from ppcls.utils import logger


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(TheseusLayer, self).__init__()
        self.res_dict = {}
        self.res_name = self.full_name()
        self.pruner = None
        self.quanter = None

    # TODO(gaotingquan): weishengyu
    def _return_dict_hook(self, layer, input, output):
        res_dict = {"output": output}
        for res_key in list(self.res_dict):
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def _save_sub_res_hook(self, layer, input, output):
        self.res_dict[self.res_name] = output

    def _find_layers_handle(self, patterns, handle_func):
        sub_layers_dict = {}
        for pattern in patterns:
            pattern_list = pattern.split(".")
            if not pattern_list:
                continue
            sub_layer_parent = self
            while len(pattern_list) > 1:
                if '[' in pattern_list[0]:
                    sub_layer_name = pattern_list[0].split('[')[0]
                    sub_layer_index = pattern_list[0].split('[')[1].split(']')[
                        0]
                    sub_layer_parent = getattr(sub_layer_parent,
                                               sub_layer_name)[sub_layer_index]
                else:
                    sub_layer_parent = getattr(sub_layer_parent,
                                               pattern_list[0], None)
                    if sub_layer_parent is None:
                        break
                if isinstance(sub_layer_parent, WrapLayer):
                    sub_layer_parent = sub_layer_parent.sub_layer
                pattern_list = pattern_list[1:]
            if sub_layer_parent is None:
                continue
            if '[' in pattern_list[0]:
                sub_layer_name = pattern_list[0].split('[')[0]
                sub_layer_index = pattern_list[0].split('[')[1].split(']')[0]
                sub_layer = getattr(sub_layer_parent,
                                    sub_layer_name)[sub_layer_index]
                if not isinstance(sub_layer, TheseusLayer):
                    sub_layer = wrap_theseus(sub_layer)
                getattr(sub_layer_parent,
                        sub_layer_name)[sub_layer_index] = sub_layer
            else:
                sub_layer = getattr(sub_layer_parent, pattern_list[0])
                if not isinstance(sub_layer, TheseusLayer):
                    sub_layer = wrap_theseus(sub_layer)
                setattr(sub_layer_parent, pattern_list[0], sub_layer)

            sub_layers_dict[pattern] = sub_layer
            handle_res = handle_func(sub_layer, pattern)
        return sub_layers_dict, handle_res

    def replace_sub(self,
                    layer_name_pattern: Union[str, List[str]],
                    replace_function: Callable[[nn.Layer, str], Any]) -> bool:
        """use 'replace_function' to modify the 'layer_name_pattern'.

        Args:
            layer_name_pattern (str): The name of target layer variable.
            replace_function (FunctionType): The function to modify target layer,

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        
        Examples:

            import paddleclas

            def replace_conv(origin_conv: nn.Conv2D):
                new_conv = nn.Conv2D(
                    in_channels=origin_conv._in_channels,
                    out_channels=origin_conv._out_channels,
                    kernel_size=origin_conv._kernel_size,
                    stride=2
                )
                return new_conv

            net = paddleclas.MobileNetV1()
            tag = net.replace_sub(layer_name_pattern="conv", replace_function=replace_conv)
            print(tag)
            # True
        """

        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]
        return self._find_layers_handle(
            layer_name_pattern, handle_func=replace_function)

    def _set_identity(self, layer, layer_name, layer_index=None):
        stop_after = False
        for sub_layer_name in layer._sub_layers:
            if stop_after:
                layer._sub_layers[sub_layer_name] = Identity()
                continue
            if sub_layer_name == layer_name:
                stop_after = True

        if layer_index and stop_after:
            stop_after = False
            for sub_layer_index in layer._sub_layers[layer_name]._sub_layers:
                if stop_after:
                    layer._sub_layers[layer_name][sub_layer_index] = Identity()
                    continue
                if layer_index == sub_layer_index:
                    stop_after = True

        return stop_after

    # stop doesn't work when stop layer has a parallel branch.
    def stop_after(self, stop_layer_name: str) -> bool:
        """stop forward and backward after 'stop_layer_name'.

        Args:
            stop_layer_name (str): The name of target layer variable.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """
        pattern_list = stop_layer_name.split(".")
        to_identity_list = []

        layer = self
        while len(pattern_list) > 0:
            layer_parent = layer
            if '[' in pattern_list[0]:
                sub_layer_name = pattern_list[0].split('[')[0]
                sub_layer_index = pattern_list[0].split('[')[1].split(']')[0]
                layer = getattr(layer, sub_layer_name)[sub_layer_index]
            else:
                sub_layer_name = pattern_list[0]
                sub_layer_index = None
                layer = getattr(layer, sub_layer_name, None)
            if layer is None:
                msg = f"Not found layer by name({pattern_list[0]}) in stop_layer_name({stop_layer_name})."
                logger.warning(msg)
                return False

            to_identity_list.append(
                (layer_parent, sub_layer_name, sub_layer_index))
            pattern_list = pattern_list[1:]

        for to_identity_layer in to_identity_list:
            if not self._set_identity(*to_identity_layer):
                msg = "Failed to set the layers that after stop_layer_name to IdentityLayer."
                logger.warning(msg)
                return False
        return True

    def update_res(self, return_patterns: Union[str, List[str]]) -> bool:
        """update the results needed returned.

        Args:
            return_patterns (Union[str, List[str]]): The layer(s)' name to be retruened.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        class Handler(object):
            def __init__(self, res_dict):
                self.res_dict = res_dict

            def __call__(self, layer, pattern):
                layer.res_dict = self.res_dict
                layer.res_name = pattern
                layer.register_forward_post_hook(layer._save_sub_res_hook)

        handle_func = Handler(self.res_dict)

        if not isinstance(return_patterns, list):
            return_patterns = [return_patterns]

        return self._find_layers_handle(
            return_patterns, handle_func=handle_func)


class WrapLayer(TheseusLayer):
    def __init__(self, sub_layer):
        super(WrapLayer, self).__init__()
        self.sub_layer = sub_layer

    def forward(self, *inputs, **kwargs):
        return self.sub_layer(*inputs, **kwargs)


def wrap_theseus(sub_layer):
    wrapped_layer = WrapLayer(sub_layer)
    return wrapped_layer
