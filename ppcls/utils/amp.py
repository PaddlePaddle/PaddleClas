import functools
import paddle


def AMP_forward_decorator(func):
    @functools.wraps(func)
    def wrapper(model, *args, **kwargs):
        if AMPForwardDecorator.amp_level:
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=AMPForwardDecorator.amp_level):
                return func(model, *args, **kwargs)
        else:
            return func(model, *args, **kwargs)

    return wrapper


class AMPForwardDecorator(object):
    amp_level = None
    amp_eval = None

    def __init__(self, forward_func):
        self.forward_func = forward_func

    @functools.wraps
    def __call__(self, model_obj, *args, **kwargs):
        # print(type(self))
        # print(type(model_obj))
        return self.forward_func(model_obj, *args, **kwargs)
