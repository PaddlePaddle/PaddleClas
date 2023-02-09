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

from .vision_transformer import ViT_base_patch16_224, ViT_small_patch16_224
from paddle import nn
import paddle
from paddle.nn.initializer import Constant, Normal, TruncatedNormal
import os

normal_ = Normal(mean=0, std=0.01)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)
trunc_normal_ = TruncatedNormal(std=.02)


class DINOHead(nn.Layer):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1D(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1D(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias_attr=False), dim=1)
        ones_(self.last_layer.weight_g)

        if norm_last_layer:
            self.last_layer.weight_g.stop_gradient = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, axis=-1, p=2)
        x = self.last_layer(x)
        return x


class DINO(nn.Layer):
    def __init__(self, **arch_config):
        super(DINO, self).__init__()
        assert arch_config['arch'] in ['ViT_small', 'ViT_base'], f"arch can be only ['ViT_small', 'ViT_base']"
        model_name = arch_config['arch'] + "_patch" + str(arch_config['patch_size']) + "_224"
        model_name = eval(model_name)
        self.train_stage = arch_config['mode']

        if arch_config['mode'] == 'pretrain':
            self.student = model_name(drop_path_rate=arch_config["drop_path_rate"])
            self.teacher = model_name()
            embed_dim = self.student.embed_dim

            # multi-crop wrapper handles forward with inputs of different resolutions
            self.student = MultiCropWrapper(
                self.student, DINOHead(
                    embed_dim,
                    arch_config["out_dim"],
                    use_bn=arch_config['use_bn_in_head'],
                    norm_last_layer=arch_config['norm_last_layer']
                )
            )
            self.teacher = MultiCropWrapper(
                self.teacher,
                DINOHead(embed_dim, arch_config["out_dim"], arch_config['use_bn_in_head'])
            )

            # vit_s8 and vit_s16 are batch norm free models. here, we don't check bn
            self.teacher = paddle.DataParallel(self.teacher)
            self.teacher_without_ddp = self.teacher._layers
            self.student = paddle.DataParallel(self.student)

            # teacher and student start with the same weights
            self.teacher_without_ddp.load_dict(self.student.state_dict())

            # there is no backpropagation through the teacher, so no need for gradients
            for p in self.teacher.parameters():
                p.stop_gradient = True

        else:
            self.model = model_name(patch_size=arch_config['patch_size'], num_classes=0)
            embed_dim = self.model.embed_dim * (arch_config['n_last_blocks'] + int(arch_config['avgpool_patchtokens']))
            logger.info(f"vit_s{arch_config['patch_size']} has build!")
            self.model.eval()

            for p in self.model.parameters():
                p.stop_gradient = True

            if os.path.isfile(arch_config['pretrained_weights']):
                state_dict = paddle.load(arch_config['pretrained_weights'])[arch_config['checkpoint_key']]
                new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                self.model.load_dict(new_state_dict)

            self.linear_clf = paddle.DataParallel(LinearClassifier(embed_dim, arch_config['num_labels']))

            self.n_last_blocks = arch_config['n_last_blocks']
            self.avgpool_patchtokens = arch_config['avgpool_patchtokens']

    def _prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.model.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.model.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)

        # add positional encoding to each token
        x = x + self.model.interpolate_pos_encoding(x, w, h)

        return self.model.pos_drop(x)

    def _get_intermediate_layers(self, x, n=1):
        x = self.model.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if len(self.model.blocks) - i <= n:
                output.append(self.model.norm(x))
        return output

    def forward(self, images):
        if self.train_stage == 'pretrain':
            teacher_out = self.teacher.forward_features(images[:2])
            student_out = self.student.forward_features(images)
            return student_out, teacher_out

        else:    # finetune
            self.linear_clf.train()

            # forward
            with paddle.no_grad():
                intermediate_output = self.get_intermediate_layers(images, self.n_last_blocks)
                output = paddle.concat([x[:, 0] for x in intermediate_output], axis=-1)
                if self.avgpool_patchtokens:
                    output = paddle.concat(
                        (output.unsqueeze(-1), paddle.mean(intermediate_output[-1][:, 1:], axis=1).unsqueeze(-1)),
                        axis=-1
                    )
                    output = output.reshape((output.shape[0], -1))

            return self.linear_clf.forward(output)


class MultiCropWrapper(nn.Layer):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]

        idx_crops = paddle.cumsum(paddle.unique_consecutive(
            paddle.to_tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx, output = 0, paddle.empty((0,))
        for end_idx in idx_crops:
            _out = self.backbone(paddle.concat(x[start_idx: end_idx]))
            if isinstance(_out, tuple):
                _out = _out[0]

            # accumulate outputs
            output = paddle.concat((output, _out))
            start_idx = end_idx

        # Run the head forward on the concatenated features.
        return self.head(output)


class LinearClassifier(nn.Layer):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        normal_(self.linear.weight)
        zeros_(self.linear.bias)

    def forward(self, x):
        # flatten
        x = x.reshape((x.shape[0], -1))

        # linear layer
        return self.linear(x)