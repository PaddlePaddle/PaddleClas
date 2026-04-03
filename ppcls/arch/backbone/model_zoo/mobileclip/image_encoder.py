# -*- coding: utf-8 -*-
from typing import Any

import paddle.nn as nn
from mobileclip import models  # Added to register models
from mobileclip.modules.image.image_projection import GlobalPool2D


class MCi(nn.Layer):
    """
    This class implements `MCi Models <https://arxiv.org/pdf/2311.17049.pdf>`_
    """

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__()
        self.projection_dim = None
        if "projection_dim" in kwargs:
            self.projection_dim = kwargs.get("projection_dim")

        # Create model
        if model_name == "mci0":
            self.model = models.mci0(projection_dim=self.projection_dim)
        elif model_name == "mci1":
            self.model = models.mci1(projection_dim=self.projection_dim)
        elif model_name == "mci2":
            self.model = models.mci2(projection_dim=self.projection_dim)
        elif model_name == "vit_b16":
            self.model = models.vit_b16(projection_dim=self.projection_dim)
        elif model_name == "vit_l14":
            self.model = models.vit_l14(projection_dim=self.projection_dim)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Build out projection head.
        if self.projection_dim is not None:
            if hasattr(self.model, "head"):
                self.model.head = MCi._update_image_classifier(
                    image_classifier=self.model.head,
                    projection_dim=self.projection_dim,
                    model=self.model,
                )

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """A forward function of the model."""
        x = self.model(x)
        return x

    @staticmethod
    def _update_image_classifier(
        image_classifier: nn.Layer, projection_dim: int, model: nn.Layer = None, *args, **kwargs
    ) -> nn.Layer:
        in_features = MCi._get_in_feature_dimension(image_classifier, model)
        new_img_classifier = GlobalPool2D(in_dim=in_features, out_dim=projection_dim)
        return new_img_classifier

    @staticmethod
    def _get_in_feature_dimension(image_classifier: nn.Layer, model: nn.Layer = None) -> int:
        """Return the input feature dimension to the image classification head."""
        in_features = None
        if isinstance(image_classifier, nn.Sequential):
            for layer in image_classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        elif isinstance(image_classifier, nn.Linear):
            in_features = image_classifier.weight.shape[0]
        elif isinstance(image_classifier, nn.Identity):
            # 针对 VisionTransformer (B/L 模型) 的处理
            # 优先从模型对象中获取 embed_dim
            if model is not None and hasattr(model, "embed_dim"):
                in_features = model.embed_dim
            else:
                # 最后的保底逻辑
                in_features = 768

        if in_features is None:
            raise NotImplementedError(
                f"Cannot get input feature dimension of {image_classifier}."
            )
        return in_features
