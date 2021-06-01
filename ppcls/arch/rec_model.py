from paddle import nn

from ppcls.arch import backbone
from ppcls.arch.gears import build_gear


class RecModel(nn.Layer):
    def __init__(self, **config):
        super().__init__()

        backbone_config = config["Backbone"]
        backbone_name = backbone_config.pop("name")
        self.backbone = getattr(backbone, backbone_name)(**backbone_config)

        assert "Stoplayer" in config, "Stoplayer should be specified in retrieval task \
                please specified a Stoplayer config"

        stop_layer_config = config["Stoplayer"]
        self.backbone.stop_after(stop_layer_config["name"])

        if stop_layer_config.get("embedding_size", 0) > 0:
            self.neck = nn.Linear(stop_layer_config["output_dim"],
                                  stop_layer_config["embedding_size"])
            embedding_size = stop_layer_config["embedding_size"]
        else:
            self.neck = None
            embedding_size = stop_layer_config["output_dim"]

        assert "Head" in config, "Head should be specified in retrieval task \
                please specify a Head config"

        config["Head"]["embedding_size"] = embedding_size
        self.head = build_gear(config["Head"])

    def forward(self, x, label):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        y = self.head(x, label)
        return {"features": x, "logits": y}
