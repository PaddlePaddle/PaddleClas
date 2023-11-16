import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..model_zoo.foundation_vit import CLIP_vit_large_patch14_224, _load_pretrained

MODEL_URLS = {
    "CLIP_large_patch14_224_aesthetic":
    "https://paddleclas.bj.bcebos.com/models/practical/pretrained/CLIP_large_patch14_224_aesthetic_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


class MLP(nn.Layer):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


class Aesthetic_Score_Predictor(nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = CLIP_vit_large_patch14_224()
        self.fc_head = nn.Linear(1024, 768, bias_attr=False)
        self.mlp = MLP(768)

    def forward(self, x):
        x = self.model(x)
        x = x[:, 0, :]
        x = self.fc_head(x)
        x = F.normalize(x, p=2, axis=-1)
        x = self.mlp(x)
        return x


def CLIP_large_patch14_224_aesthetic(pretrained=False,
                                     use_ssld=False,
                                     **kwargs):
    model = Aesthetic_Score_Predictor()
    _load_pretrained(pretrained, model,
                     MODEL_URLS["CLIP_large_patch14_224_aesthetic"], use_ssld)
    return model
