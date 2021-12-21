from typing import Tuple, List, Dict, Union, Callable, Any
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

    def replace_sub(self, *args, **kwargs) -> None:
        msg = "\"replace_sub\" is deprecated, please use \"layer_wrench\" instead."
        logger.error(DeprecationWarning(msg))
        raise DeprecationWarning(msg)

    # TODO(gaotingquan): what is a good name?
    def layer_wrench(self,
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
            # {'blocks[11].depthwise_conv.conv': the corresponding new_layer, 'blocks[12].depthwise_conv.conv': the corresponding new_layer}
        """

        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]

        handle_res_dict = {}
        for pattern in layer_name_pattern:
            # find parent layer of sub-layer specified by pattern
            sub_layer_parent = None
            for target_layer_dict in parse_pattern_str(
                    pattern=pattern, idx=(0, -1), parent_layer=self):
                sub_layer_parent = target_layer_dict["target_layer"]

            if not sub_layer_parent:
                continue

            # find sub-layer specified by pattern
            sub_layer = None
            for target_layer_dict in parse_pattern_str(
                    pattern=pattern, idx=-1, parent_layer=sub_layer_parent):
                sub_layer = target_layer_dict["target_layer"]
                sub_layer_name = target_layer_dict["target_layer_name"]
                sub_layer_index = target_layer_dict["target_layer_index"]

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

    def stop_after(self, stop_layer_name: str) -> bool:
        """stop forward and backward after 'stop_layer_name'.

        Args:
            stop_layer_name (str): The name of layer that stop forward and backward after this layer.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        to_identity_list = []

        for target_layer_dict in parse_pattern_str(stop_layer_name, self):
            sub_layer_name = target_layer_dict["target_layer_name"]
            sub_layer_index = target_layer_dict["target_layer_index"]
            parent_layer = target_layer_dict["parent_layer"]
            to_identity_list.append(
                (parent_layer, sub_layer_name, sub_layer_index))

        for to_identity_layer in to_identity_list:
            if not set_identity(*to_identity_layer):
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


def set_identity(parent_layer: nn.Layer,
                 layer_name: str,
                 layer_index: str=None) -> bool:
    """set the layer specified by layer_name and layer_index to Indentity.

    Args:
        parent_layer (nn.Layer): The parent layer of target layer specified by layer_name and layer_index.
        layer_name (str): The name of target layer to be set to Indentity.
        layer_index (str, optional): The index of target layer to be set to Indentity in parent_layer. Defaults to None.

    Returns:
        bool: True if successfully, False otherwise.
    """

    stop_after = False
    for sub_layer_name in parent_layer._sub_layers:
        if stop_after:
            parent_layer._sub_layers[sub_layer_name] = Identity()
            continue
        if sub_layer_name == layer_name:
            stop_after = True

    if layer_index and stop_after:
        stop_after = False
        for sub_layer_index in parent_layer._sub_layers[
                layer_name]._sub_layers:
            if stop_after:
                parent_layer._sub_layers[layer_name][
                    sub_layer_index] = Identity()
                continue
            if layer_index == sub_layer_index:
                stop_after = True

    return stop_after


def slice_pattern(pattern: str, idx: Union[Tuple, int]=None) -> List:
    """slice the string type "pattern" to list type by separator ".".

    Args:
        pattern (str): The pattern to discribe layer name.
        idx (Union[Tuple, int], optional): The index(s) of sub-list of list sliced. Defaults to None.

    Returns:
        List: The sub-list of list sliced by "pattern".
    """

    pattern_list = pattern.split(".")
    if idx:
        if isinstance(idx, Tuple):
            if len(idx) == 1:
                return pattern_list[idx[0]]
            elif len(idx) == 2:
                return pattern_list[idx[0]:idx[1]]
            else:
                msg = f"Only support length of 'idx' is 1 or 2 when 'idx' is a Tuple."
                logger.warning(msg)
                return None
        elif isinstance(idx, int):
            return [pattern_list[idx]]
        else:
            msg = f"Only support type of 'idx' is int or Tuple."
            logger.warning(msg)
            return None

    return pattern_list


def parse_pattern_str(pattern: str, parent_layer: nn.Layer,
                      idx=None) -> Dict[str, Union[nn.Layer, None, str]]:
    """parse the string type pattern.

    Args:
        pattern (str): The pattern to discribe layer name.
        parent_layer (nn.Layer): The parent layer of target layer(s) specified by "pattern".
        idx ([type], optional): [description]. The index(s) of sub-list of list sliced. Defaults to None.

    Returns:
        Dict[str, Union[nn.Layer, None, str]]: Dict["target_layer": Union[nn.Layer, None], "target_layer_name": str, "target_layer_index": str, "parent_layer": nn.Layer]

    Yields:
        Iterator[Dict[str, Union[nn.Layer, None, str]]]: Dict["target_layer": Union[nn.Layer, None], "target_layer_name": str, "target_layer_index": str, "parent_layer": nn.Layer]
    """

    pattern_list = slice_pattern(pattern, idx)
    if not pattern_list:
        return None, None, None

    while len(pattern_list) > 0:
        if '[' in pattern_list[0]:
            target_layer_name = pattern_list[0].split('[')[0]
            target_layer_index = pattern_list[0].split('[')[1].split(']')[0]
        else:
            target_layer_name = pattern_list[0]
            target_layer_index = None

        target_layer = getattr(parent_layer, target_layer_name, None)
        target_layer = unwrap_theseus(target_layer)

        if target_layer is None:
            msg = f"Not found layer named({target_layer_name}) specifed in pattern({pattern})."
            logger.warning(msg)
            return {
                "target_layer": None,
                "target_layer_name": target_layer_name,
                "target_layer_index": target_layer_index,
                "parent_layer": parent_layer
            }

        if target_layer_index and target_layer:
            if int(target_layer_index) < 0 or int(target_layer_index) >= len(
                    target_layer):
                msg = f"Not found layer by index({target_layer_index}) specifed in pattern({pattern}). The lenght of sub_layer's parent layer is < '{len(parent_layer)}' and > '0'."
                logger.warning(msg)
                return {
                    "target_layer": None,
                    "target_layer_name": target_layer_name,
                    "target_layer_index": target_layer_index,
                    "parent_layer": parent_layer
                }
            target_layer = target_layer[target_layer_index]
            target_layer = unwrap_theseus(target_layer)

        yield {
            "target_layer": target_layer,
            "target_layer_name": target_layer_name,
            "target_layer_index": target_layer_index,
            "parent_layer": parent_layer
        }

        pattern_list = pattern_list[1:]
        parent_layer = target_layer
