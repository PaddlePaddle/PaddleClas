# -*- coding: utf-8 -*-
from typing import Dict
import paddle
from paddle import nn

try:
    import open_clip
except Exception:
    open_clip = None

class ClipTokenizer(nn.Layer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        # MobileCLIP2 config has a different structure
        text_cfg = cfg.get("text_cfg", cfg)
        self.context_length = text_cfg["context_length"]
        
        if open_clip is not None:
            model_name = text_cfg.get("open_clip_tokenizer", "ViT-B-16")
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            self.tokenizer = None
            print("Warning: open_clip not found, tokenizer will not work properly.")

    def get_vocab_size(self) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encoder)
        return 49408

    def forward(self, input_sentence: str, *args, **kwargs) -> paddle.Tensor:
        if self.tokenizer:
            tokenized_sentence = self.tokenizer(input_sentence, self.context_length)
            return paddle.to_tensor(tokenized_sentence)
        else:
            return paddle.zeros([1, self.context_length], dtype="int64")
