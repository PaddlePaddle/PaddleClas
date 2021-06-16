# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ""))
sys.path.append(os.path.join(__dir__, "deploy"))

import argparse
import shutil
import textwrap
import tarfile
import requests
import warnings
from functools import partial
from difflib import SequenceMatcher

import cv2
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

from deploy.python.predict_cls import ClsPredictor
from deploy.utils.get_image_list import get_image_list
from deploy.utils import config

from ppcls.arch.backbone import *

__all__ = ["PaddleClas"]

BASE_DIR = os.path.expanduser("~/.paddleclas/")
BASE_INFERENCE_MODEL_DIR = os.path.join(BASE_DIR, "inference_model")
BASE_IMAGES_DIR = os.path.join(BASE_DIR, "images")
BASE_DOWNLOAD_URL = "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/{}_infer.tar"
MODEL_SERIES = {
    "AlexNet": ["AlexNet"],
    "DarkNet": ["DarkNet53"],
    "DeiT": [
        "DeiT_base_distilled_patch16_224", "DeiT_base_distilled_patch16_384",
        "DeiT_base_patch16_224", "DeiT_base_patch16_384",
        "DeiT_small_distilled_patch16_224", "DeiT_small_patch16_224",
        "DeiT_tiny_distilled_patch16_224", "DeiT_tiny_patch16_224"
    ],
    "DenseNet": [
        "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",
        "DenseNet264"
    ],
    "DPN": ["DPN68", "DPN92", "DPN98", "DPN107", "DPN131"],
    "EfficientNet": [
        "EfficientNetB0", "EfficientNetB0_small", "EfficientNetB1",
        "EfficientNetB2", "EfficientNetB3", "EfficientNetB4", "EfficientNetB5",
        "EfficientNetB6", "EfficientNetB7"
    ],
    "GhostNet":
    ["GhostNet_x0_5", "GhostNet_x1_0", "GhostNet_x1_3", "GhostNet_x1_3_ssld"],
    "HRNet": [
        "HRNet_W18_C", "HRNet_W30_C", "HRNet_W32_C", "HRNet_W40_C",
        "HRNet_W44_C", "HRNet_W48_C", "HRNet_W64_C", "HRNet_W18_C_ssld",
        "HRNet_W48_C_ssld"
    ],
    "Inception": ["GoogLeNet", "InceptionV3", "InceptionV4"],
    "MobileNetV1": [
        "MobileNetV1_x0_25", "MobileNetV1_x0_5", "MobileNetV1_x0_75",
        "MobileNetV1", "MobileNetV1_ssld"
    ],
    "MobileNetV2": [
        "MobileNetV2_x0_25", "MobileNetV2_x0_5", "MobileNetV2_x0_75",
        "MobileNetV2", "MobileNetV2_x1_5", "MobileNetV2_x2_0",
        "MobileNetV2_ssld"
    ],
    "MobileNetV3": [
        "MobileNetV3_small_x0_35", "MobileNetV3_small_x0_5",
        "MobileNetV3_small_x0_75", "MobileNetV3_small_x1_0",
        "MobileNetV3_small_x1_25", "MobileNetV3_large_x0_35",
        "MobileNetV3_large_x0_5", "MobileNetV3_large_x0_75",
        "MobileNetV3_large_x1_0", "MobileNetV3_large_x1_25",
        "MobileNetV3_small_x1_0_ssld", "MobileNetV3_large_x1_0_ssld"
    ],
    "RegNet": ["RegNetX_4GF"],
    "Res2Net": [
        "Res2Net50_14w_8s", "Res2Net50_26w_4s", "Res2Net50_vd_26w_4s",
        "Res2Net200_vd_26w_4s", "Res2Net101_vd_26w_4s",
        "Res2Net50_vd_26w_4s_ssld", "Res2Net101_vd_26w_4s_ssld",
        "Res2Net200_vd_26w_4s_ssld"
    ],
    "ResNeSt": ["ResNeSt50", "ResNeSt50_fast_1s1x64d"],
    "ResNet": [
        "ResNet18", "ResNet18_vd", "ResNet34", "ResNet34_vd", "ResNet50",
        "ResNet50_vc", "ResNet50_vd", "ResNet50_vd_v2", "ResNet101",
        "ResNet101_vd", "ResNet152", "ResNet152_vd", "ResNet200_vd",
        "ResNet34_vd_ssld", "ResNet50_vd_ssld", "ResNet50_vd_ssld_v2",
        "ResNet101_vd_ssld", "Fix_ResNet50_vd_ssld_v2", "ResNet50_ACNet_deploy"
    ],
    "ResNeXt": [
        "ResNeXt50_32x4d", "ResNeXt50_vd_32x4d", "ResNeXt50_64x4d",
        "ResNeXt50_vd_64x4d", "ResNeXt101_32x4d", "ResNeXt101_vd_32x4d",
        "ResNeXt101_32x8d_wsl", "ResNeXt101_32x16d_wsl",
        "ResNeXt101_32x32d_wsl", "ResNeXt101_32x48d_wsl",
        "Fix_ResNeXt101_32x48d_wsl", "ResNeXt101_64x4d", "ResNeXt101_vd_64x4d",
        "ResNeXt152_32x4d", "ResNeXt152_vd_32x4d", "ResNeXt152_64x4d",
        "ResNeXt152_vd_64x4d"
    ],
    "SENet": [
        "SENet154_vd", "SE_HRNet_W64_C_ssld", "SE_ResNet18_vd",
        "SE_ResNet34_vd", "SE_ResNet50_vd", "SE_ResNeXt50_32x4d",
        "SE_ResNeXt50_vd_32x4d", "SE_ResNeXt101_32x4d"
    ],
    "ShuffleNetV2": [
        "ShuffleNetV2_swish", "ShuffleNetV2_x0_25", "ShuffleNetV2_x0_33",
        "ShuffleNetV2_x0_5", "ShuffleNetV2_x1_0", "ShuffleNetV2_x1_5",
        "ShuffleNetV2_x2_0"
    ],
    "SqueezeNet": ["SqueezeNet1_0", "SqueezeNet1_1"],
    "SwinTransformer": [
        "SwinTransformer_large_patch4_window7_224_22kto1k",
        "SwinTransformer_large_patch4_window12_384_22kto1k",
        "SwinTransformer_base_patch4_window7_224_22kto1k",
        "SwinTransformer_base_patch4_window12_384_22kto1k",
        "SwinTransformer_base_patch4_window12_384",
        "SwinTransformer_base_patch4_window7_224",
        "SwinTransformer_small_patch4_window7_224",
        "SwinTransformer_tiny_patch4_window7_224"
    ],
    "VGG": ["VGG11", "VGG13", "VGG16", "VGG19"],
    "VisionTransformer": [
        "ViT_base_patch16_224", "ViT_base_patch16_384", "ViT_base_patch32_384",
        "ViT_large_patch16_224", "ViT_large_patch16_384",
        "ViT_large_patch32_384", "ViT_small_patch16_224"
    ],
    "Xception": [
        "Xception41", "Xception41_deeplab", "Xception65", "Xception65_deeplab",
        "Xception71"
    ]
}


class ImageTypeError(Exception):
    """ImageTypeError.
    """

    def __init__(self, message=""):
        super().__init__(message)


class InputModelError(Exception):
    """InputModelError.
    """

    def __init__(self, message=""):
        super().__init__(message)


def args_cfg():
    parser = config.parser()
    other_options = [
        ("infer_imgs", str, None, "The image(s) to be predicted."),
        ("model_name", str, None, "The model name to be used."),
        ("inference_model_dir", str, None, "The directory of model files."),
        ("use_gpu", bool, True, "Whether use GPU. Default by True."), (
            "enable_mkldnn", bool, False,
            "Whether use MKLDNN. Default by False."),
        ("batch_size", int, 1, "Batch size. Default by 1.")
    ]
    for name, opt_type, default, description in other_options:
        parser.add_argument(
            "--" + name, type=opt_type, default=default, help=description)

    args = parser.parse_args()

    for name, opt_type, default, description in other_options:
        val = eval("args." + name)
        full_name = "Global." + name
        args.override.append(
            f"{full_name}={val}") if val is not default else None

    cfg = config.get_config(
        args.config, overrides=args.override, show=args.verbose)

    return cfg


def get_default_confg():
    return {
        "Global": {
            "model_name": "MobileNetV3_small_x0_35",
            "use_gpu": False,
            "use_fp16": False,
            "enable_mkldnn": False,
            "cpu_num_threads": 1,
            "use_tensorrt": False,
            "ir_optim": False,
            "enable_profile": False
        },
        "PreProcess": {
            "transform_ops": [{
                "ResizeImage": {
                    "resize_short": 256
                }
            }, {
                "CropImage": {
                    "size": 224
                }
            }, {
                "NormalizeImage": {
                    "scale": 0.00392157,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "order": ""
                }
            }, {
                "ToCHWImage": None
            }]
        },
        "PostProcess": {
            "name": "Topk",
            "topk": 5,
            "class_id_map_file": "./ppcls/utils/imagenet1k_label_list.txt"
        }
    }


def print_info():
    """Print list of supported models in formatted.
    """
    table = PrettyTable(["Series", "Name"])
    try:
        sz = os.get_terminal_size()
        width = sz.columns - 30 if sz.columns > 50 else 10
    except OSError:
        width = 100
    for series in MODEL_SERIES:
        names = textwrap.fill("  ".join(MODEL_SERIES[series]), width=width)
        table.add_row([series, names])
    width = len(str(table).split("\n")[0])
    print("{}".format("-" * width))
    print("Models supported by PaddleClas".center(width))
    print(table)
    print("Powered by PaddlePaddle!".rjust(width))
    print("{}".format("-" * width))


def get_model_names():
    """Get the model names list.
    """
    model_names = []
    for series in MODEL_SERIES:
        model_names += (MODEL_SERIES[series])
    return model_names


def similar_architectures(name="", names=[], thresh=0.1, topk=10):
    """Find the most similar topk model names.
    """
    scores = []
    for idx, n in enumerate(names):
        if n.startswith("__"):
            continue
        score = SequenceMatcher(None, n.lower(), name.lower()).quick_ratio()
        if score > thresh:
            scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    similar_names = [names[s[0]] for s in scores[:min(topk, len(scores))]]
    return similar_names


def download_with_progressbar(url, save_path):
    """Download from url with progressbar.
    """
    if os.path.isfile(save_path):
        os.remove(save_path)
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes or not os.path.isfile(
            save_path):
        raise Exception(
            f"Something went wrong while downloading model/image from {url}")

def check_model_file(model_name):
    """Check the model files exist and download and untar when no exist. 
    """
    storage_directory = partial(os.path.join, BASE_INFERENCE_MODEL_DIR,
                                model_name)
    url = BASE_DOWNLOAD_URL.format(model_name)

def check_model_file(model_name):
    """Check the model files exist and download and untar when no exist. 
    """
    storage_directory = partial(os.path.join, BASE_INFERENCE_MODEL_DIR,
                                model_name)
    url = BASE_DOWNLOAD_URL.format(model_name)

    tar_file_name_list = [
        "inference.pdiparams", "inference.pdiparams.info", "inference.pdmodel"
    ]
    model_file_path = storage_directory("inference.pdmodel")
    params_file_path = storage_directory("inference.pdiparams")
    if not os.path.exists(model_file_path) or not os.path.exists(
            params_file_path):
        tmp_path = storage_directory(url.split("/")[-1])
        print(f"download {url} to {tmp_path}")
        os.makedirs(storage_directory(), exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, "r") as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if tar_file_name in member.name:
                        filename = tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(storage_directory(filename), "wb") as f:
                    f.write(file.read())
        os.remove(tmp_path)
    if not os.path.exists(model_file_path) or not os.path.exists(
            params_file_path):
        raise Exception(
            f"Something went wrong while praparing the model[{model_name}] files!"
        )

    return storage_directory()


def save_prelabel_results(class_id, input_file_path, output_dir):
    """Save the predicted image according to the prediction.
    """
    output_dir = os.path.join(output_dir, str(class_id))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(input_file_path, output_dir)


class PaddleClas(object):
    """PaddleClas.
    """

    print_info()

    def __init__(self,
                 config: dict=None,
                 model_name: str=None,
                 inference_model_dir: str=None,
                 use_gpu: bool=None,
                 batch_size: int=None):
        """Init PaddleClas with config.

        Args:
            config: The config of PaddleClas's predictor, default by None. If default, the default configuration is used. Please refer doc for more information.
            model_name: The model name supported by PaddleClas, default by None. If specified, override config.
            inference_model_dir: The directory that contained model file and params file to be used, default by None. If specified, override config.
            use_gpu: Wheather use GPU, default by None. If specified, override config.
            batch_size: The batch size to pridict, default by None. If specified, override config.
        """
        super().__init__()
        self._config = config
        self._check_config(model_name, inference_model_dir, use_gpu,
                           batch_size)
        self._check_input_model()
        self.cls_predictor = ClsPredictor(self._config)

    def get_config(self):
        """Get the config.
        """
        return self._config

    def _check_config(self,
                      model_name=None,
                      inference_model_dir=None,
                      use_gpu=None,
                      batch_size=None):
        if self._config is None:
            self._config = get_default_confg()
            warnings.warn("config is not provided, use default!")
        self._config = config.AttrDict(self._config)
        config.create_attr_dict(self._config)

        if model_name is not None:
            self._config.Global["model_name"] = model_name
        if inference_model_dir is not None:
            self._config.Global["inference_model_dir"] = inference_model_dir
        if use_gpu is not None:
            self._config.Global["use_gpu"] = use_gpu
        if batch_size is not None:
            self._config.Global["batch_size"] = batch_size

    def _check_input_model(self):
        """Check input model name or model files.
        """
        candidate_model_names = get_model_names()
        input_model_name = self._config.Global.get("model_name", None)
        inference_model_dir = self._config.Global.get("inference_model_dir",
                                                      None)
        if input_model_name is not None:
            similar_names = similar_architectures(input_model_name,
                                                  candidate_model_names)
            similar_names_str = ", ".join(similar_names)
            if input_model_name not in similar_names_str:
                err = f"{input_model_name} is not exist! Maybe you want: [{similar_names_str}]"
                raise InputModelError(err)
            if input_model_name not in candidate_model_names:
                err = f"{input_model_name} is not provided by PaddleClas. If you want to use your own model, please input model_file as model path!"
                raise InputModelError(err)
            self._config.Global.inference_model_dir = check_model_file(
                input_model_name)
            return
        elif inference_model_dir is not None:
            model_file_path = os.path.join(inference_model_dir,
                                           "inference.pdmodel")
            params_file_path = os.path.join(inference_model_dir,
                                            "inference.pdiparams")
            if not os.path.isfile(model_file_path) or not os.path.isfile(
                    params_file_path):
                err = f"There is no model file or params file in this directory: {inference_model_dir}"
                raise InputModelError(err)
            return
        else:
            err = f"Please specify the model name supported by PaddleClas or directory contained model file and params file."
            raise InputModelError(err)
        return

    def predict(self, input_data, print_pred=True):
        """Predict label of img with paddleclas.
        Args:
            input_data(str, NumPy.ndarray): 
                image to be classified, support: str(local path of image file, internet URL, directory containing series of images) and NumPy.ndarray(preprocessed image data that has 3 channels and accords with [C, H, W], or raw image data that has 3 channels and accords with [H, W, C]).
        Returns:
            dict: {image_name: "", class_id: [], scores: [], label_names: []}，if label name path == None，label_names will be empty.
        """
        if isinstance(input_data, np.ndarray):
            return self.cls_predictor.predict(input_data)
        elif isinstance(input_data, str):
            if input_data.startswith("http"):
                image_storage_dir = partial(os.path.join, BASE_IMAGES_DIR)
                if not os.path.exists(image_storage_dir()):
                    os.makedirs(image_storage_dir())
                image_save_path = image_storage_dir("tmp.jpg")
                download_with_progressbar(input_data, image_save_path)
                input_data = image_save_path
                warnings.warn(
                    f"Image to be predicted from Internet: {input_data}, has been saved to: {image_save_path}"
                )
            image_list = get_image_list(input_data)

            batch_size = self._config.Global.get("batch_size", 1)
            pre_label_out_idr = self._config.Global.get("pre_label_out_idr",
                                                        False)

            img_list = []
            img_path_list = []
            output_list = []
            cnt = 0
            for idx, img_path in enumerate(image_list):
                img = cv2.imread(img_path)
                if img is None:
                    warnings.warn(
                        f"Image file failed to read and has been skipped. The path: {img_path}"
                    )
                    continue
                img_list.append(img)
                img_path_list.append(img_path)
                cnt += 1

                if cnt % batch_size == 0 or (idx + 1) == len(image_list):
                    outputs = self.cls_predictor.predict(img_list)
                    output_list.append(outputs[0])
                    preds = self.cls_predictor.postprocess(outputs)
                    for nu, pred in enumerate(preds):
                        if pre_label_out_idr:
                            save_prelabel_results(pred["class_ids"][0],
                                                  img_path_list[nu],
                                                  pre_label_out_idr)
                        if print_pred:
                            pred_str_list = [
                                f"filename: {img_path_list[nu]}",
                                f"top-{self._config.PostProcess.get('topk', 1)}"
                            ]
                            for k in pred:
                                pred_str_list.append(f"{k}: {pred[k]}")
                            print(", ".join(pred_str_list))
                    img_list = []
                    img_path_list = []
            return output_list
        else:
            err = "Please input legal image! The type of image supported by PaddleClas are: NumPy.ndarray and string of local path or Ineternet URL"
            raise ImageTypeError(err)
        return


# for CLI
def main():
    """Function API used for commad line.
    """
    cfg = args_cfg()
    clas_engine = PaddleClas(cfg)
    clas_engine.predict(cfg["Global"]["infer_imgs"], print_pred=True)
    return


if __name__ == "__main__":
    main()
