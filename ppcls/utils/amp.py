from functools import partial
import contextlib
import paddle


class AutoCast:
    def __init__(self,
                 use_amp=False,
                 amp_level="O1",
                 use_promote=False,
                 amp_eval=False):
        self.use_amp = use_amp
        self.amp_eval = amp_eval

        if self.use_amp:
            # compatible with paddle 2.5 and older version
            paddle_version = paddle.__version__[:3]
            # paddle version >= 2.5.0 or develop
            if paddle_version in ["2.5", "0.0"]:
                self.cast_context = partial(
                    paddle.amp.auto_cast,
                    level=amp_level,
                    use_promote=use_promote)
            # paddle version <= 2.4.x and not develop
            else:
                self.cast_context = partial(
                    paddle.amp.auto_cast, level=amp_level)

    def __call__(self, is_eval=False):
        if self.use_amp:
            # not is_eval: cast for all training
            # is_eval and self.amp_eval: cast for evaluation only when amp_eval is True
            if not is_eval or (is_eval and self.amp_eval):
                return self.cast_context()

        return contextlib.nullcontext()


def build_scaler(use_amp=False, scale_loss=1.0,
                 use_dynamic_loss_scaling=False):
    class Foo:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            return

        def minimize(self, optimizer, loss):
            optimizer.step()

    if use_amp:
        return paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling)
    return Foo()
