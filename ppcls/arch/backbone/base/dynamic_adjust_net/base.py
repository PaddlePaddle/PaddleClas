from ..theseus_layer import TheseusLayer


class DynamicAdjustNet(TheseusLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.net(x)

    def post_iter(self):
        raise NotImplementedError

    def post_epoch(self):
        raise NotImplementedError
