import paddle
import paddle.nn as nn
from ..model_zoo.efficientnet import EfficientNetB3, _load_pretrained

MODEL_URLS = {
    "EfficientNetB3_watermark":
    "https://paddleclas.bj.bcebos.com/models/practical/pretrained/EfficientNetB3_watermark_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


def EfficientNetB3_watermark(padding_type='DYNAMIC',
                             override_params={"batch_norm_epsilon": 0.00001},
                             use_se=True,
                             pretrained=False,
                             use_ssld=False,
                             **kwargs):
    def replace_function(_fc, pattern):
        classifier = nn.Sequential(
            # 1536 is the orginal in_features
            nn.Linear(
                in_features=1536, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(
                in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(
                in_features=256, out_features=2), )
        return classifier

    pattern = "_fc"
    model = EfficientNetB3(
        padding_type=padding_type,
        override_params=override_params,
        use_se=True,
        pretrained=False,
        use_ssld=False,
        **kwargs)
    model.upgrade_sublayer(pattern, replace_function)
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientNetB3_watermark"],
                     use_ssld)
    return model
