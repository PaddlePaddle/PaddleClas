# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was heavily based on https://github.com/facebookresearch/deit
# reference: https://arxiv.org/abs/2012.12877

import math
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal

from .modeling_finetune import VisionTransformer, zeros_, ones_
from .norm_ema_quantizer import NormEMAVectorQuantizer

MODEL_URLS = {
    "vqkd":
    "vqkd.pdparams",
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=.02)
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class VQKD(nn.Layer):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192, 
                 embed_dim=32,
                 decay=0.99,
                 process_type='default',
                 quantize_kmeans_init=True,
                 teacher_model_type='clip',
                 decoder_out_dim=512,
                 rec_loss_type='cosine',
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config['in_chans'] != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim
        
        # encoder & decode params
        print('Final encoder config', encoder_config)
        self.encoder = VisionTransformer(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder = VisionTransformer(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )

        self.patch_size = encoder_config['patch_size']
        self.token_shape = (encoder_config['img_size'] // self.patch_size, encoder_config['img_size'] // self.patch_size)
       
        ## Teacher model setting
        self.teacher_model_type = teacher_model_type
        self.decoder_out_dim = decoder_out_dim
        self.teacher_model = None
        # if self.teacher_model_type == 'clip':
        #     self.scaling_layer = ScalingLayerForClip()
        #     self.teacher_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
        #     self.decoder_out_dim = 512

        # elif self.teacher_model_type == 'dino':
        #     self.scaling_layer = ScalingLayerForIM()
        #     self.teacher_model = get_dino_vit_base()
        #     self.decoder_out_dim = 768

        # else:
        #     self.teacher_model = None

        # if self.teacher_model is not None:
        #     for param in self.teacher_model.parameters():
        #         param.requires_grad = False # fix teacher_model model

        #     self.teacher_model.eval()
        #     self.teacher_input_size = kwargs.get('teacher_input_size', 224)

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim) # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        
        self.rec_loss_type = rec_loss_type

        print(f"process type for VQKD: {process_type}")
        self.process_type = process_type # in ['default', 'dall-e']
        self.logit_laplace_eps = 0.1
        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 
                'encoder.cls_token', 'encoder.pos_embed'}

    def device(self):
        return self.decoder.cls_token.device

    def pre_process(self, data):
        if self.process_type == 'default':
            # TODO: modify for adapt
            if data.max() <= 1.:
                data = data * 255.
            data = data / 127.5 - 1.0
        elif self.process_type == 'imagenet_norm':
            mean = paddle.to_tensor(IMAGENET_DEFAULT_MEAN).set_device(self.device)[None, :, None, None]
            std = paddle.to_tensor(IMAGENET_DEFAULT_STD).set_device(self.device)[None, :, None, None]
            data = (data - mean) / std
        return data

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, **kwargs):
        data = self.pre_process(data)
        quantize, embed_ind, loss = self.encode(data)
        output = {}
        output['token'] = embed_ind.reshape([data.shape[0], -1])
        output['input_img'] = data

        return output

    def encode(self, x):
        encoder_features = self.encoder(x, return_patch_tokens=True)

        with paddle.amp.auto_cast(enable=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.astype(self.encode_task_layer[-1].weight.dtype))
        # to_quantizer_features = self.encode_task_layer(encoder_features.astype(self.encode_task_layer[-1].weight.dtype))

        N = to_quantizer_features.shape[1]
        h, w = int(math.sqrt(N)), int(math.sqrt(N))

        b, c = to_quantizer_features.shape[0], to_quantizer_features.shape[-1]
        to_quantizer_features = paddle.reshape(to_quantizer_features, [b, c, h, w])
        # to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss

    def decode(self, quantize, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        # quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=self.token_shape[0], w=self.token_shape[1])
        decoder_features = self.decoder(quantize, return_patch_tokens=True)
        rec = self.decode_task_layer(decoder_features)

        return rec

    def get_codebook_indices(self, x, **kwargs):
        # for beit pre-training
        return self.get_tokens(x, **kwargs)['token']

    def get_regress_target(self, x, **kwargs):

        norm_imgs = self.scaling_layer(x)
        if self.teacher_model_type == 'clip':
            target = self.teacher_model.encode_image(norm_imgs, return_all_tokens=True) @ self.teacher_model.visual.proj
        elif self.teacher_model_type == 'dino':
            target = self.teacher_model.forward(norm_imgs, return_patch_tokens=True)
        else:
            raise NotImplementedError

        return target
    
    def calculate_rec_loss(self, rec, target):  
        if self.rec_loss_type == 'cosine':
            target = target / target.norm(dim=-1, keepdim=True)
            rec = rec / rec.norm(dim=-1, keepdim=True)
            rec_loss = (1 - (target * rec).sum(-1)).mean()
        else:
            raise NotImplementedError

        return rec_loss

    def forward(self, x, **kwargs):
        """
        x: shape [B, 3, H, W] in [0, 1]
        """
        x = self.pre_process(x) # rescale to [-1, 1]

        target = self.get_regress_target(x, **kwargs)

        quantize, embed_ind, emb_loss = self.encode(x)
        xrec = self.decode(quantize)

        rec_loss = self.calculate_rec_loss(xrec, target)
        loss = emb_loss + rec_loss

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_loss'] = rec_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log


class ScalingLayerForClip(nn.Layer):
    def __init__(self):
        super(ScalingLayerForClip, self).__init__()
        self.register_buffer('shift', paddle.to_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None])
        self.register_buffer('scale', paddle.to_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None])

    def forward(self, inp):
        inp = ((inp + 1.) * 127.5).clamp(0, 255.) / 255. # rescale to [0, 1.]
        return (inp - self.shift) / self.scale

class ScalingLayerForIM(nn.Layer):
    def __init__(self):
        super(ScalingLayerForIM, self).__init__()
        self.register_buffer('shift', paddle.to_tensor([0.485, 0.456, 0.406])[None, :, None, None]) # scale for tokenizer with default prosscess type \in [-1, 1]
        self.register_buffer('scale', paddle.to_tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, inp):
        inp = ((inp + 1.) * 127.5).clamp(0, 255.) / 255. # rescale to [0, 1.]
        return (inp - self.shift) / self.scale


def get_model_default_params():
    return dict(img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,  
                             mlp_ratio=4., qkv_bias=True,  qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                             norm_layer="nn.LayerNorm", init_values=0., use_abs_pos_emb=True, 
                             use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


def vqkd_encoder_base_decoder_3x768x12_clip(pretrained=False, pretrained_weight=None, as_tokenzer=False, img_size=224, 
                                            n_code=8192, code_dim=32, **kwargs):
    encoder_config, decoder_config = get_model_default_params(), get_model_default_params()
    
    # encoder settings
    encoder_config['img_size'] = img_size
    encoder_config['num_classes'] = 0
    # decoder settings
    decoder_config['img_size'] = img_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    # teacher settings
    _ = kwargs.pop("teacher_model_type", "clip")
    
    # teacher_model_type = 'clip' if not as_tokenzer else 'None'
    teacher_model_type = 'None'
    decoder_out_dim = 512

    model = VQKD(encoder_config,
                 decoder_config,
                 n_code,
                 code_dim,
                 teacher_model_type=teacher_model_type,
                 decoder_out_dim=decoder_out_dim,
                 **kwargs)

    # if as_tokenzer:
    #     assert pretrained
    #     assert pretrained_weight is not None

    #     if pretrained_weight.startswith('https'):
    #         weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
    #     else:
    #         weights = torch.load(pretrained_weight, map_location='cpu')
        
    #     if 'model' in weights:
    #         weights = weights['model']
    #     else:
    #         weights = weights["state_dict"]
    #     keys = list(weights.keys())
        
    #     for k in keys:
    #         if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
    #             del weights[k]
    #     model.load_state_dict(weights)
    weight = paddle.load(pretrained_weight)
    model.set_dict(weight)
    return model