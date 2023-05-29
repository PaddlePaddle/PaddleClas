import os.path as osp
from typing import Tuple, Union

import paddle.distributed as dist
import numpy as np
import paddle
from paddle.nn import functional as F
from paddle import nn
from paddle.static.nn import sequence_pad as pad_sequence

from .utils import trunc_normal_, interpolate_pos_embed
from .VL_LTR_pretrain import ModifiedResNet, VisionTransformer
from paddle.nn.initializer import Assign, Normal, Constant,TruncatedNormal

__all__ = [
            'LGR_r50', 
            'LGR_vit16',
        ]



class QuickGELU(nn.Layer):
    def forward(self, x: paddle.Tensor):
        return x * F.sigmoid(1.702 * x)

class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.norm1q = nn.LayerNorm(dim)
        self.norm1k = nn.LayerNorm(dim)

        self.wq = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, qx: paddle.Tensor, kx: paddle.Tensor, key_padding_mask: paddle.Tensor = None):
        # qx: [Bq, 1, C]    kx: [Bk, Nk, C]
        # key_padding_mask: [Bk, Nk] (mask==1 ==> '-inf')
        # output: [Bq, Bk, C]
        assert qx.shape[-1] == kx.shape[-1] and qx.shape[1] == 1
        Bq, _, C = qx.shape
        Bk, Nk, _ = kx.shape
        q = self.wq(self.norm1q(qx))
        q = paddle.reshape(q,(Bq, 1, self.num_heads, C // self.num_heads))
        q = paddle.transpose(q,(0, 2, 1, 3))

        k = self.wq(self.norm1k(kx))
        k = paddle.reshape(k,(Bk, Nk, self.num_heads, C // self.num_heads))
        k = paddle.transpose(k,(0, 2, 1, 3))

        #k = self.wk(self.norm1k(kx)).reshape(Bk, Nk, self.num_heads, C //
        #                                     self.num_heads).permute(0, 2, 1, 3)
        
        v = paddle.unsqueeze(kx,axis=1)
        #v = kx.unsqueeze(1)
        #  q: [Bq, num_heads,  1, C // num_heads]
        # kv: [Bk, num_heads, Nk, C // num_heads]
        # attn: [Bq, Bk, num_heads, Nk]
        attn = paddle.einsum('qhoc,khnc->qkhn', q, k) * self.scale
        if key_padding_mask is not None:
            attn = masked_fill(attn, paddle.unsqueeze(paddle.unsqueeze(key_padding_mask,axis=0),axis=2),float('-inf'))
        attn = F.softmax(attn,axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.einsum('khnc,qkhn->qkhc', v, attn)
        x = paddle.reshape(x,(Bq, Bk, C))

        return x


class Block(nn.Layer):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 op_type='two_branch', num_classes=0, use_constant_norm=False, v_detach=False):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.op_type = op_type
        self.use_constant_norm = use_constant_norm
        self.v_detach = v_detach
        if self.op_type == 'concat':
            self.fc = nn.Linear(in_features=dim * 2, out_features=1, bias=True)
        elif self.op_type == 'add':
            self.fc = nn.Linear(in_features=dim, out_features=1, bias=True)
        elif self.op_type == 'cosine':
            self.fc = None
        elif self.op_type == 'two_branch':
            self.cos = nn.CosineSimilarity(axis=2, eps=1e-6)
            self.visual_fc = nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.ReLU(),
                nn.Linear(4 * dim, num_classes))
        else:
            self.fc = None

    def forward(self, qx: paddle.Tensor, kx: paddle.Tensor, key_padding_mask: paddle.Tensor = None, logit_scale=None):
        # qx: [Bq, 1, C]    kx: [Bk, Nk, C]
        # v: [Bq, Bk, C]
        v = self.attn(qx, kx, key_padding_mask=key_padding_mask)
        if self.op_type == 'concat':
            x = paddle.expand(qx,(qx.shape[0], kx.shape[0], qx.shape[-1]))
            #x = qx.expand(qx.shape[0], kx.shape[0], qx.shape[-1])
            x = paddle.concat((x,v),axis=-1)
            #x = torch.cat((x, v), dim=-1)  # [Bq, Bk, 2*C]
            x = self.fc(x)  # [Bq, Bk, 1]
        elif self.op_type == 'cosine':
            if logit_scale is not None:
                qx_ = F.normalize(qx, p=2, axis=-1)
                if self.v_detach:
                    v_buff = paddle.linalg.norm(v,axis=-1,keepdim=True).detach()
                    v_ = v /v_buff
                else:
                    v_ = F.normalize(v, p=2, axis=-1)
                x = paddle.einsum('qkc,qoc->qk', v_, qx_) * paddle.exp(logit_scale) 
            else:
                x = paddle.einsum('qkc,qoc->qk', v, qx)
        elif self.op_type == 'add':
            x = paddle.expand(qx,(qx.shape[0], kx.shape[0], qx.shape[-1]))
            x = x + v
            x = self.fc(x)  # [Bq, Bk, 1]
        elif self.op_type == 'two_branch':
            x1 = self.visual_fc(paddle.squeeze(qx,axis=1))

            if logit_scale is not None:
                if self.use_constant_norm:
                    qx_ = F.normalize(qx, p=2, axis=-1)
                    v_ = v / 21.1578
                    x2 = paddle.einsum('qkc,qoc->qk', v_, qx_) *  paddle.exp(logit_scale) 
                else:
                    qx_ = F.normalize(qx, p=2, axis=-1)
                    if self.v_detach:
                        v_buff = paddle.linalg.norm(v,axis=-1,keepdim=True).detach()
                        v_ = v / v_buff
                    else:
                        v_ = F.normalize(v, p=2, axis=-1)
                    x2 = paddle.einsum('qkc,qoc->qk', v_, qx_) *  paddle.exp(logit_scale) 
            else:
                x2 = paddle.einsum('qkc,qoc->qk', v, qx)
            

            return x1, x2

        return paddle.squeeze(x,axis=-1) 


class LGR(nn.Layer):
    def __init__(self,
                 num_classes: int,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 sent_length: int,
                 attn_heads: int,
                 sent_idxs=None,
                 op_type="two_branch",
                 use_norm=False,
                 use_constant_norm=False,
                 v_detach=False,
                 img_grad=True,
                 attn_grad=True,
                 select_sent=None,
                 sent_offset=0,
                 ):
        super().__init__()
        self.num_classes = num_classes

        self.sent_offset = sent_offset
        self.sent_length = sent_length
        self.sent_idxs = sent_idxs
        self.select_sent = select_sent

        self.use_norm = use_norm
        self.img_grad = img_grad
        self.attn_grad = attn_grad

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers, output_dim=embed_dim,
                heads=vision_heads, input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width, layers=vision_layers,
                heads=vision_heads, output_dim=embed_dim
            )

        if op_type is None:
            print("do not use text features")
            self.text_embeddings = None
            self.text_block = None
            self.text_padding_mask = None
            self.fc = nn.Linear(embed_dim, num_classes)
        else:
            self.fc = None
            if self.use_norm:
                self.logit_scale = self.create_parameter(
                    (1,),
                    default_initializer=Assign(paddle.ones([1])* np.log(1 / 0.07))
                )
                #self.logit_scale = paddle.ones([]) * np.log(1 / 0.07)
                self.logit_scale.stop_gradient = True
            else:
                self.logit_scale = None
            self.text_embeddings = self.create_parameter(
                    (self.num_classes, self.sent_length, embed_dim)
                )
            #self.text_embeddings = paddle.empty(self.num_classes, self.sent_length, embed_dim)
            
            self.text_block = Block(dim=embed_dim, num_heads=attn_heads,
                                    qkv_bias=False, qk_scale=None, drop=0,
                                    attn_drop=0,
                                    op_type=op_type, num_classes=num_classes,
                                    use_constant_norm=use_constant_norm,
                                    v_detach=v_detach
                                    )
            self.text_padding_mask = self.build_key_padding_mask(paddle.to_tensor(self.sent_idxs))

        #self.initialize_parameters()

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer freezed."""
        super(LGR, self).train()
        if mode:
            if self.img_grad is False:
                print('freeze visual norm')
                for m in self.visual.parameters():
                    if isinstance(m, nn.LayerNorm):
                        m.eval()
                    if isinstance(m, nn.BatchNorm2D):
                        m.eval()
            if self.attn_grad is False:
                print('freeze attn norm')
                for m in self.text_block.attn.parameters():
                    if isinstance(m, nn.LayerNorm):
                        m.eval()
                    if isinstance(m, nn.BatchNorm2D):
                        m.eval()
        self.initialize_parameters()

    def to(self, device, *args):
        super().to(device=device, *args)
        if self.text_padding_mask is not None:
            self.text_padding_mask = self.text_padding_mask.to(device=device, *args)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def _init_weights(self, m):
        
        if isinstance(m, nn.Linear):
            TruncatedNormal(std=0.02)(m.weight)
            #m.weight.set_value(paddle.trunc(paddle.normal(std=0.02)))
            if isinstance(m, nn.Linear) and m.bias is not None:
                #nn.init.constant_(m.bias, 0)
                Constant(0.0)(m.bias)
                #m.bias.set_value(paddle.zeros_like(m.bias))

        elif isinstance(m, nn.LayerNorm):
            #nn.init.constant_(m.bias, 0)
            Constant(0.0)(m.bias)
            #m.bias.set_value(paddle.zeros_like(m.bias))
            #nn.init.constant_(m.weight, 1.0)
            Constant(1.0)(m.weight)
            #m.weight.set_value(paddle.ones_like(m.bias))
        if self.text_block is not None:
            Assign(paddle.eye(self.text_block.attn.wq.weight.shape[0]))(self.text_block.attn.wq.weight)
            Assign(paddle.eye(self.text_block.attn.wk.weight.shape[0]))(self.text_block.attn.wk.weight)
            #self.text_block.attn.wq.weight.set_value(paddle.eye(self.text_block.attn.wq.weight.shape[0]))
            #self.text_block.attn.wk.weight.set_value(paddle.eye(self.text_block.attn.wk.weight.shape[0]))
            #nn.init.eye_(self.text_block.attn.wq.weight)
            #nn.init.eye_(self.text_block.attn.wk.weight)
        
    def initialize_parameters(self):
        self.apply(self._init_weights)

    def build_key_padding_mask(self, idxs: paddle.Tensor):
        # 根据idxs生成mask，>idxs的位置设为True，以防止pad 0的影响
        mask = paddle.arange(0,self.sent_length)
        mask = paddle.cast(mask,idxs.dtype)
        mask = paddle.unsqueeze(mask,axis=0)
        #mask = torch.arange(0, self.sent_length).type_as(idxs).unsqueeze(0)
        mask = paddle.expand(mask,(idxs.shape[0], self.sent_length))
        mask = paddle.greater_than(mask, paddle.unsqueeze(idxs,axis=1)-1)
        #mask = mask.expand(idxs.shape[0], self.sent_length).gt(idxs.unsqueeze(1) - 1)
        return mask

    def load_pretrained_model(self, txt_embed_path=None, vis_backbone_path=None, img_grad=True,
                              attn_grad=True):
        if txt_embed_path is not None:
            self._load_text_embeddings(txt_embed_path)
            self.text_embeddings.stop_gradient = True
        if vis_backbone_path is not None:
            self._load_vis_backbone(vis_backbone_path)
            self.visual.stop_gradient = not img_grad
            self.text_block.attn.stop_gradient = not attn_grad

    def _load_text_embeddings(self, txt_embed_path):
        assert self.text_embeddings is not None
        if not osp.exists(txt_embed_path):
            print("warning: no txt embeddings found, please generate the txt embeddings first in the pretraining stage")
            return
        text_embeddings = paddle.to_tensor(np.load(txt_embed_path))  # [Nt, embed_dim]
        split_text_embeddings = paddle.split(text_embeddings, self.sent_idxs)
        
        if self.select_sent == 'rand':
            print('randomly selecting sents')
            split_text_embeddings = [s[self.sent_offset:, :] for s in split_text_embeddings]
            split_sorted_idxs = [paddle.randperm(len(s))[:self.sent_length] for s in split_text_embeddings]
            split_text_embeddings = [s[i] for s,i in zip(split_text_embeddings, split_sorted_idxs)]
        elif self.select_sent is not None:
            print('using selected sents')
            txt_ces_path = txt_embed_path.replace('txt_embed.npy', '%s_txt_ce.npy' % self.select_sent)
            assert osp.exists(txt_ces_path)
            text_ces = paddle.to_tensor(np.load(txt_ces_path))  # [Nt, embed_dim]
            split_text_ces = paddle.split(text_ces, self.sent_idxs)
            split_sorted_idxs = [paddle.sort(text_ces)[1][:self.sent_length] for text_ces in split_text_ces]
            split_text_embeddings = [text_embeddings[sorted_idxs] for sorted_idxs, text_embeddings in
                                        zip(split_sorted_idxs, split_text_embeddings)]
        else:
            split_text_embeddings = [s[self.sent_offset:self.sent_length+self.sent_offset, :] for s in split_text_embeddings]
        if self.select_sent != "rand":
            split_text_embeddings = pad_sequence(split_text_embeddings,pad_value=0.0)
        self.text_embeddings.data = split_text_embeddings
        print("text embeddings loaded")


    def _load_vis_backbone(self, vis_backbone_path):
        if vis_backbone_path.endswith('RN50.pt') or \
            vis_backbone_path.endswith('ViT-B-32.pt') or \
            vis_backbone_path.endswith('ViT-B-16.pt'):
            pretrained_state_dict = paddle.load(
                vis_backbone_path).state_dict()
        else:
            pretrained_state_dict = paddle.load(
                vis_backbone_path)
        
        if isinstance(self.visual, VisionTransformer):
            num_extra_tokens = 1
            new_size = int((self.visual.positional_embedding.shape[0] - num_extra_tokens) ** 0.5)
            new_pos_embed = interpolate_pos_embed(pretrained_state_dict['visual.positional_embedding'], 
                                                    new_size, num_extra_tokens=num_extra_tokens)
            pretrained_state_dict['visual.positional_embedding'] = new_pos_embed
        if self.use_norm:
            vis_state_dict = {
                k: v for k, v in pretrained_state_dict.items()
                if k.startswith("visual") or k.startswith('logit_scale')
            }
        else:
            vis_state_dict = {
                k: v for k, v in pretrained_state_dict.items()
                if k.startswith("visual")
            }

        info = self.set_state_dict(vis_state_dict)
        print('pretrained visual backbone loaded')
        print(info)


    def encode_image(self, image) -> paddle.Tensor:
        self.visual.eval()
        x = self.visual(paddle.cast(image,self.dtype))
        #x = self.visual(image.type(self.dtype))
        return x

    def forward(self, x):
        x = self.encode_image(x)
        if self.text_block is not None:

            x = self.text_block( paddle.unsqueeze(x,axis=1), paddle.cast(self.text_embeddings,x.dtype),
                                key_padding_mask=self.text_padding_mask,
                                logit_scale=self.logit_scale)
        else:   
            x = self.fc(x)
        return x


def LGR_r50_test_api(pretrained=False, **kwargs):
    args = kwargs['args']
    dataset = kwargs['dataset']
    sent_idxs = getattr(dataset, 'end_idxs', [0])

    model = LGR(
        num_classes=args.nb_classes,
        embed_dim=1024,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=None,
        sent_length=args.sent_length,
        attn_heads=1,
        sent_idxs=sent_idxs,
        use_norm=True,
        img_grad=False,
        select_sent='rand'
    )

    vis_backbone_path = osp.join(args.pretrain_cvlp_path, "checkpoint.pdparams")
    if not osp.exists(vis_backbone_path):
        print("no ckpt file found")
        vis_backbone_path = args.pretrained_clip
    model.load_pretrained_model(
        txt_embed_path=osp.join(args.pretrain_cvlp_path, "txt_embed.npy"),
        vis_backbone_path=vis_backbone_path, img_grad=False
    )

    return model
def LGR_r50(pretrained=False, args=None,dataset=None):
    args = args
    dataset = dataset
    sent_idxs = getattr(dataset, 'end_idxs', None)

    model = LGR(
        num_classes=args.nb_classes,
        embed_dim=1024,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=None,
        sent_length=args.sent_length,
        attn_heads=1,
        sent_idxs=sent_idxs,
        use_norm=True,
        img_grad=False,
        select_sent='rand'
    )

    vis_backbone_path = osp.join(args.pretrain_cvlp_path, "checkpoint.pth")
    if not osp.exists(vis_backbone_path):
        print("no ckpt file found")
        vis_backbone_path = args.pretrained_clip
    model.load_pretrained_model(
        txt_embed_path=osp.join(args.pretrain_cvlp_path, "txt_embed.npy"),
        vis_backbone_path=vis_backbone_path, img_grad=False
    )

    return model



def LGR_vit16(pretrained=False, **kwargs):
    args = kwargs['args']
    dataset = kwargs['dataset']
    sent_idxs = getattr(dataset, 'end_idxs', None)
    #select_sent = 'val' if args.test else 'train'

    model = LGR(
        num_classes=args.nb_classes,
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        sent_length=args.sent_length,
        attn_heads=1,
        sent_idxs=sent_idxs,
        use_norm=True,
        img_grad=False,
        select_sent="rand"
    )

    model.load_pretrained_model(
        txt_embed_path=os.join(args.pretrain_cvlp_path, "txt_embed.npy"),
        vis_backbone_path=os.join(args.pretrain_cvlp_path, "checkpoint.pth"),
        img_grad=False
    )

    return model
