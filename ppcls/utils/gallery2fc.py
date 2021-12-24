import os
import paddle
import cv2

from ppcls.arch import build_model
from ppcls.arch.gears.identity_head import IdentityHead
from ppcls.utils.config import parse_config, parse_args
from ppcls.utils.save_load import load_dygraph_pretrain
from ppcls.utils.logger import init_logger
from ppcls.data import transform, create_operators
from ppcls.arch.slim import quantize_model


class GalleryLayer(paddle.nn.Layer):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        embedding_size = self.configs["Arch"]["Head"]["embedding_size"]
        self.batch_size = self.configs["IndexProcess"]["batch_size"]
        self.image_shape = self.configs["Global"]["image_shape"].copy()
        self.image_shape.insert(0, self.batch_size)

        image_root = self.configs["IndexProcess"]["image_root"]
        data_file = self.configs["IndexProcess"]["data_file"]
        delimiter = self.configs["IndexProcess"]["delimiter"]
        self.gallery_images = []
        gallery_docs = []
        gallery_labels = []

        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for ori_line in lines:
                line = ori_line.strip().split(delimiter)
                text_num = len(line)
                assert text_num >= 2, f"line({ori_line}) must be splitted into at least 2 parts, but got {text_num}"
                image_file = os.path.join(image_root, line[0])

                self.gallery_images.append(image_file)
                gallery_docs.append(ori_line.strip())
                gallery_labels.append(line[1].strip())
        self.gallery_layer = paddle.nn.Linear(embedding_size, len(self.gallery_images), bias_attr=False)

    def forward(self, x, label=None):
        x = paddle.nn.functional.normalize(x)
        x = self.gallery_layer(x)
        return x

    def build_gallery_layer(self, feature_extractor):
        transform_configs = self.configs["IndexProcess"]["transform_ops"]
        preprocess_ops = create_operators(transform_configs)
        embedding_size = self.configs["Arch"]["Head"]["embedding_size"]
        batch_index = 0
        input_tensor = paddle.zeros(self.image_shape)
        gallery_feature = paddle.zeros((len(self.gallery_images), embedding_size))
        for i, image_path in enumerate(self.gallery_images):
            image = cv2.imread(image_path)
            for op in preprocess_ops:
                image = op(image)
            input_tensor[batch_index] = image
            batch_index += 1
            if batch_index == self.batch_size or i == len(self.gallery_images) - 1:
                batch_feature = feature_extractor(input_tensor)["features"]
                for j in range(batch_index):
                    feature = batch_feature[j]
                    norm_feature = paddle.nn.functional.normalize(feature, axis=0)
                    gallery_feature[i - batch_index + j] = norm_feature
        self.gallery_layer.set_state_dict({"weight": gallery_feature.T})


def export_fuse_model(configs):
    slim_config = configs["Slim"].copy()
    configs["Slim"] = None
    fuse_model = build_model(configs)
    fuse_model.head = GalleryLayer(configs)
    configs["Slim"] = slim_config
    quantize_model(configs, fuse_model)
    load_dygraph_pretrain(fuse_model, configs["Global"]["pretrained_model"])
    fuse_model.eval()
    fuse_model.head.build_gallery_layer(fuse_model)
    save_path = configs["Global"]["save_inference_dir"]
    fuse_model.quanter.save_quantized_model(
        fuse_model,
        save_path,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + configs["Global"]["image_shape"],
                dtype='float32')
        ])


def main():
    args = parse_args()
    configs = parse_config(args.config)
    init_logger(name='gallery2fc')
    export_fuse_model(configs)


if __name__ == '__main__':
    main()
