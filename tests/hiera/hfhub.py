import inspect


class PyTorchModelHubMixin:
    error_str: str = (
        'This feature requires "huggingface-hub >= 0.21.0" to be installed.'
    )

    @classmethod
    def from_pretrained(cls, *args, **kwdargs):
        raise RuntimeError(cls.error_str)

    @classmethod
    def save_pretrained(cls, *args, **kwdargs):
        raise RuntimeError(cls.error_str)

    @classmethod
    def push_to_hub(cls, *args, **kwdargs):
        raise RuntimeError(cls.error_str)


def has_config(func):
    signature = inspect.signature(func)

    def wrapper(self, *args, **kwdargs):
        if "config" in kwdargs:
            config = kwdargs["config"]
            del kwdargs["config"]
            kwdargs.update(**config)
        self.config = {
            k: (v.default if i - 1 >= len(args) else args[i - 1])
            for i, (k, v) in enumerate(signature.parameters.items())
            if v.default is not inspect.Parameter.empty
        }
        self.config.update(**kwdargs)
        func(self, **kwdargs)

    return wrapper
