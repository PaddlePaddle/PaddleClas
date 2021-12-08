import paddle
from ppcls.arch import build_model
from ppcls.utils.config import parse_config, parse_args


def build_gallery_feature(feature_extractor):
    pass


def save_fuse_model(fuse_model):
    pass


class FuseModel(paddle.nn.Layer):
    def __init__(self, configs):
        super().__init__()
        self.feature_extractor = build_model(configs)
        self.gallery_layer = build_gallery_feature(self.feature_extractor)

    def forward(self, x):
        x = self.feature_model(x)
        x = self.gallery_layer(x)
        return x


def main():
    args = parse_args()
    configs = parse_config(args.config)
    fuse_model = FuseModel(configs)
    save_fuse_model(fuse_model)


if __name__ == '__main__':
    main()
