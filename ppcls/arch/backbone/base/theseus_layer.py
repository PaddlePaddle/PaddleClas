from typing import List, Dict, Union, Callable, Any
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

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"output": output}
        # 'list' is needed to avoid error raised by popping self.res_dict
        for res_key in list(self.res_dict):
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def _save_sub_res_hook(self, layer, input, output):
        self.res_dict[self.res_name] = output

    def replace_sub(self,
                    layer_name_pattern: Union[str, List[str]],
                    handle_func: Callable[[nn.Layer, str], nn.Layer]) -> Dict[
                        str, nn.Layer]:
        """use 'handle_func' to modify the sub-layer(s) specified by 'layer_name_pattern'.

        Args:
            layer_name_pattern (Union[str, List[str]]): The name of layer to be modified by 'handle_func'.
            handle_func (Callable[[nn.Layer, str], nn.Layer]): The function to modify target layer specified by 'layer_name_pattern'.

        Returns:
            Dict[str, nn.Layer]: The key is the patter and corresponding value is the result returned by 'handle_func'.
        
        Examples:

            from paddle import nn
            import paddleclas

            def rep_func(sub_layer: nn.Layer, pattern: str):
                new_layer = nn.Conv2D(
                    in_channels=sub_layer._in_channels,
                    out_channels=sub_layer._out_channels,
                    kernel_size=5,
                    padding=2
                )
                return new_layer

            net = paddleclas.MobileNetV1()
            res = net.replace_sub(layer_name_pattern=["blocks[11].depthwise_conv.conv", "blocks[12].depthwise_conv.conv"], handle_func=rep_func)
            print(res)
            # {'blocks[11].depthwise_conv.conv': True, 'blocks[12].depthwise_conv.conv': True}
        """
        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]

        handle_res_dict = {}
        for pattern in layer_name_pattern:
            # pattern_list = pattern.split(".")

            # find parent layer of sub-layer specified by pattern
            sub_layer_parent, _, _ = parse_pattern_str(
                pattern=pattern, idx=(0, -1), sub_layer_parent=self)

            if not sub_layer_parent:
                continue

            # find sub-layer specified by pattern
            sub_layer, sub_layer_name, sub_layer_index = parse_pattern_str(
                pattern=pattern, idx=-1, sub_layer_parent=sub_layer_parent)

            if not sub_layer:
                continue

            new_sub_layer = handle_func(sub_layer, pattern)

            if sub_layer_index:
                getattr(sub_layer_parent,
                        sub_layer_name)[sub_layer_index] = new_sub_layer
            else:
                setattr(sub_layer_parent, sub_layer_name, new_sub_layer)

            handle_res_dict[pattern] = new_sub_layer
        return handle_res_dict

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

    def stop_after(self, stop_layer_name: str) -> bool:
        """stop forward and backward after 'stop_layer_name'.

        Args:
            stop_layer_name (str): The name of layer that stop forward and backward after this layer.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """
        pattern_list = stop_layer_name.split(".")
        to_identity_list = []

        # TODO(gaotingquan): replace code by self._parse_pattern_str()
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
                msg = f"Not found layer by name({pattern_list[0]}) specifed in stop_layer_name({stop_layer_name})."
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

    def update_res(self,
                   return_patterns: Union[str, List[str]]) -> Dict[str, bool]:
        """update the results to be returned.

        Args:
            return_patterns (Union[str, List[str]]): The name of layer to return output.

        Returns:
            Dict[str, bool]: The pattern(str) is be set successfully if 'True'(bool), failed if 'False'(bool).
        """

        class Handler(object):
            def __init__(self, res_dict):
                self.res_dict = res_dict

            def __call__(self, layer, pattern):
                layer.res_dict = self.res_dict
                layer.res_name = pattern
                layer.register_forward_post_hook(layer._save_sub_res_hook)
                return layer

        handle_func = Handler(self.res_dict)

        return self.replace_sub(return_patterns, handle_func=handle_func)


class WrapLayer(TheseusLayer):
    def __init__(self, sub_layer):
        super(WrapLayer, self).__init__()
        self.sub_layer = sub_layer

    def forward(self, *inputs, **kwargs):
        return self.sub_layer(*inputs, **kwargs)


def wrap_theseus(sub_layer):
    return WrapLayer(sub_layer)


def unwrap_theseus(sub_layer):
    if isinstance(sub_layer, WrapLayer):
        sub_layer = sub_layer.sub_layer
    return sub_layer


def slice_pattern(pattern, idx):
    pattern_list = pattern.split(".")
    if idx:
        if isinstance(idx, tuple):
            if len(idx) == 1:
                return pattern_list[idx[0]]
            elif len(idx) == 2:
                return pattern_list[idx[0]:idx[1]]
            else:
                msg = f"Only support length of 'idx' is 1 or 2 when 'idx' is a tuple."
                logger.warning(msg)
                return None
        elif isinstance(idx, int):
            return [pattern_list[idx]]
        else:
            msg = f"Only support type of 'idx' is int or tuple."
            logger.warning(msg)
            return None

    return pattern_list


def parse_pattern_str(pattern, sub_layer_parent, idx=None):
    pattern_list = slice_pattern(pattern, idx)
    if not pattern_list:
        return None, None, None

    while len(pattern_list) > 0:
        if '[' in pattern_list[0]:
            sub_layer_name = pattern_list[0].split('[')[0]
            sub_layer_index = pattern_list[0].split('[')[1].split(']')[0]
        else:
            sub_layer_name = pattern_list[0]
            sub_layer_index = None

        sub_layer_parent = getattr(sub_layer_parent, sub_layer_name, None)
        sub_layer_parent = unwrap_theseus(sub_layer_parent)

        if sub_layer_parent is None:
            msg = f"Not found layer named({sub_layer_name}) specifed in pattern({pattern})."
            logger.warning(msg)
            return None, sub_layer_name, sub_layer_index

        if sub_layer_index and sub_layer_parent:
            if int(sub_layer_index) < 0 or int(sub_layer_index) >= len(
                    sub_layer_parent):
                msg = f"Not found layer by index({sub_layer_index}) specifed in pattern({pattern}). The lenght of sub_layer's parent layer is < '{len(sub_layer_parent)}' and > '0'."
                logger.warning(msg)
                return None, sub_layer_name, sub_layer_index
            sub_layer_parent = sub_layer_parent[sub_layer_index]
            sub_layer_parent = unwrap_theseus(sub_layer_parent)

        pattern_list = pattern_list[1:]

    return sub_layer_parent, sub_layer_name, sub_layer_index
