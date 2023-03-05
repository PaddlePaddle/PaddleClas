def clas_forward_decorator(forward_func):
    def parse_batch_wrapper(model, batch):
        x, label = batch[0], batch[1]
        return forward_func(model, x)

    return parse_batch_wrapper
