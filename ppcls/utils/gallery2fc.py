import os
import paddle

from ppcls.arch import build_model
from ppcls.arch.gears.identity_head import IdentityHead
from ppcls.utils.config import parse_config, parse_args
from ppcls.utils.save_load import load_dygraph_pretrain
from ppcls.utils.logger import init_logger
from ppcls.data import transform, create_operators


def build_gallery_feature(configs, feature_extractor):
    transform_configs = configs["Infer"]["transforms"]
    preprocess_ops = create_operators(transform_configs)

    embedding_size = configs["Arch"]["Head"]["embedding_size"]
    batch_size = configs["IndexProcess"]["batch_size"]
    image_shape = configs["Global"]["image_shape"]
    image_shape.insert(0, batch_size)
    input_tensor = paddle.zeros(image_shape)

    image_root = configs["IndexProcess"]["image_root"]
    data_file = configs["IndexProcess"]["data_file"]
    delimiter = configs["IndexProcess"]["delimiter"]
    gallery_images = []
    gallery_docs = []

    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _, ori_line in enumerate(lines):
            line = ori_line.strip().split(delimiter)
            text_num = len(line)
            assert text_num >= 2, f"line({ori_line}) must be splitted into at least 2 parts, but got {text_num}"
            image_file = os.path.join(image_root, line[0])

            gallery_images.append(image_file)
            gallery_docs.append(ori_line.strip())





def save_fuse_model(fuse_model):
    pass


class FuseModel(paddle.nn.Layer):
    def __init__(self, configs):
        super().__init__()
        self.feature_extractor = build_model(configs)
        load_dygraph_pretrain(self.feature_extractor, configs["Global"]["pretrained_model"])
        self.feature_extractor.head = IdentityHead()
        self.gallery_layer = build_gallery_feature(configs, self.feature_extractor)

    def forward(self, x):
        x = self.feature_model(x)["features"]
        x = paddle.norm(x)
        x = self.gallery_layer(x)
        return x


def main():
    args = parse_args()
    configs = parse_config(args.config)
    init_logger(name='gallery2fc')
    fuse_model = FuseModel(configs)
    save_fuse_model(fuse_model)


if __name__ == '__main__':
    main()
