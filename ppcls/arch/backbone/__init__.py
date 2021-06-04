# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from ppcls.arch.backbone.legendary_models.mobilenet_v1 import MobileNetV1_x0_25, MobileNetV1_x0_5, MobileNetV1_x0_75, MobileNetV1
from ppcls.arch.backbone.legendary_models.mobilenet_v3 import MobileNetV3_small_x0_35, MobileNetV3_small_x0_5, MobileNetV3_small_x0_75, MobileNetV3_small_x1_0, MobileNetV3_small_x1_25, MobileNetV3_large_x0_35, MobileNetV3_large_x0_5, MobileNetV3_large_x0_75, MobileNetV3_large_x1_0, MobileNetV3_large_x1_25
from ppcls.arch.backbone.legendary_models.resnet import ResNet18, ResNet18_vd, ResNet34, ResNet34_vd, ResNet50, ResNet50_vd, ResNet101, ResNet101_vd, ResNet152, ResNet152_vd, ResNet200_vd
from ppcls.arch.backbone.legendary_models.vgg import VGG11, VGG13, VGG16, VGG19
from ppcls.arch.backbone.legendary_models.inception_v3 import InceptionV3
from ppcls.arch.backbone.legendary_models.hrnet import HRNet_W18_C, HRNet_W30_C, HRNet_W32_C, HRNet_W40_C, HRNet_W44_C, HRNet_W48_C, HRNet_W60_C, HRNet_W64_C, SE_HRNet_W64_C

from ppcls.arch.backbone.model_zoo.resnet_vc import ResNet18_vc, ResNet34_vc, ResNet50_vc, ResNet101_vc, ResNet152_vc
from ppcls.arch.backbone.model_zoo.resnext import ResNeXt50_32x4d, ResNeXt50_64x4d, ResNeXt101_32x4d, ResNeXt101_64x4d, ResNeXt152_32x4d, ResNeXt152_64x4d
from ppcls.arch.backbone.model_zoo.res2net import Res2Net50_48w_2s, Res2Net50_26w_4s, Res2Net50_14w_8s, Res2Net50_48w_2s, Res2Net50_26w_6s, Res2Net50_26w_8s, Res2Net101_26w_4s, Res2Net152_26w_4s, Res2Net200_26w_4s
from ppcls.arch.backbone.model_zoo.res2net_vd import Res2Net50_vd_48w_2s, Res2Net50_vd_26w_4s, Res2Net50_vd_14w_8s, Res2Net50_vd_48w_2s, Res2Net50_vd_26w_6s, Res2Net50_vd_26w_8s, Res2Net101_vd_26w_4s, Res2Net152_vd_26w_4s, Res2Net200_vd_26w_4s
from ppcls.arch.backbone.model_zoo.se_resnet_vd import SE_ResNet18_vd, SE_ResNet34_vd, SE_ResNet50_vd, SE_ResNet101_vd, SE_ResNet152_vd, SE_ResNet200_vd
from ppcls.arch.backbone.model_zoo.se_resnext_vd import SE_ResNeXt50_vd_32x4d, SE_ResNeXt50_vd_32x4d, SENet154_vd
from ppcls.arch.backbone.model_zoo.se_resnext import SE_ResNeXt50_32x4d, SE_ResNeXt101_32x4d, SE_ResNeXt152_64x4d
from ppcls.arch.backbone.model_zoo.dpn import DPN68, DPN92, DPN98, DPN107, DPN131
from ppcls.arch.backbone.model_zoo.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, DenseNet264
from ppcls.arch.backbone.model_zoo.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetB0_small
from ppcls.arch.backbone.model_zoo.resnest import ResNeSt50_fast_1s1x64d, ResNeSt50, ResNeSt101
from ppcls.arch.backbone.model_zoo.googlenet import GoogLeNet
from ppcls.arch.backbone.model_zoo.mobilenet_v2 import MobileNetV2_x0_25, MobileNetV2_x0_5, MobileNetV2_x0_75, MobileNetV2, MobileNetV2_x1_5, MobileNetV2_x2_0
from ppcls.arch.backbone.model_zoo.shufflenet_v2 import ShuffleNetV2_x0_25, ShuffleNetV2_x0_33, ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0, ShuffleNetV2_swish
from ppcls.arch.backbone.model_zoo.alexnet import AlexNet
from ppcls.arch.backbone.model_zoo.inception_v4 import InceptionV4
from ppcls.arch.backbone.model_zoo.xception import Xception41, Xception65, Xception71
from ppcls.arch.backbone.model_zoo.xception_deeplab import Xception41_deeplab, Xception65_deeplab, Xception71_deeplab
from ppcls.arch.backbone.model_zoo.resnext101_wsl import ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl, ResNeXt101_32x32d_wsl, ResNeXt101_32x48d_wsl
from ppcls.arch.backbone.model_zoo.squeezenet import SqueezeNet1_0, SqueezeNet1_1
from ppcls.arch.backbone.model_zoo.darknet import DarkNet53
from ppcls.arch.backbone.model_zoo.regnet import RegNetX_200MF, RegNetX_4GF, RegNetX_32GF, RegNetY_200MF, RegNetY_4GF, RegNetY_32GF
from ppcls.arch.backbone.model_zoo.vision_transformer import ViT_small_patch16_224, ViT_base_patch16_224, ViT_base_patch16_384, ViT_base_patch32_384, ViT_large_patch16_224, ViT_large_patch16_384, ViT_large_patch32_384, ViT_huge_patch16_224, ViT_huge_patch32_384
from ppcls.arch.backbone.model_zoo.distilled_vision_transformer import DeiT_tiny_patch16_224, DeiT_small_patch16_224, DeiT_base_patch16_224, DeiT_tiny_distilled_patch16_224, DeiT_small_distilled_patch16_224, DeiT_base_distilled_patch16_224, DeiT_base_patch16_384, DeiT_base_distilled_patch16_384
from ppcls.arch.backbone.model_zoo.distillation_models import ResNet50_vd_distill_MobileNetV3_large_x1_0, ResNeXt101_32x16d_wsl_distill_ResNet50_vd
from ppcls.arch.backbone.model_zoo.swin_transformer import SwinTransformer_tiny_patch4_window7_224, SwinTransformer_small_patch4_window7_224, SwinTransformer_base_patch4_window7_224, SwinTransformer_base_patch4_window12_384, SwinTransformer_large_patch4_window7_224, SwinTransformer_large_patch4_window12_384
from ppcls.arch.backbone.model_zoo.mixnet import MixNet_S, MixNet_M, MixNet_L
from ppcls.arch.backbone.model_zoo.rexnet import ReXNet_1_0, ReXNet_1_3, ReXNet_1_5, ReXNet_2_0, ReXNet_3_0
from ppcls.arch.backbone.variant_models.resnet_variant import ResNet50_last_stage_stride1
