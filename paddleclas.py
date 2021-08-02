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

from typing import Union, Generator
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


def init_config(model_name,
                inference_model_dir,
                use_gpu=True,
                batch_size=1,
                topk=5,
                **kwargs):
    imagenet1k_map_path = os.path.join(
        os.path.abspath(__dir__), "ppcls/utils/imagenet1k_label_list.txt")
    cfg = {
        "Global": {
            "infer_imgs": kwargs["infer_imgs"]
            if "infer_imgs" in kwargs else False,
            "model_name": model_name,
            "inference_model_dir": inference_model_dir,
            "batch_size": batch_size,
            "use_gpu": use_gpu,
            "enable_mkldnn": kwargs["enable_mkldnn"]
            if "enable_mkldnn" in kwargs else False,
            "cpu_num_threads": kwargs["cpu_num_threads"]
            if "cpu_num_threads" in kwargs else 1,
            "enable_benchmark": False,
            "use_fp16": kwargs["use_fp16"] if "use_fp16" in kwargs else False,
            "ir_optim": True,
            "use_tensorrt": kwargs["use_tensorrt"]
            if "use_tensorrt" in kwargs else False,
            "gpu_mem": kwargs["gpu_mem"] if "gpu_mem" in kwargs else 8000,
            "enable_profile": False
        },
        "PreProcess": {
            "transform_ops": [{
                "ResizeImage": {
                    "resize_short": kwargs["resize_short"]
                    if "resize_short" in kwargs else 256
                }
            }, {
                "CropImage": {
                    "size": kwargs["crop_size"]
                    if "crop_size" in kwargs else 224
                }
            }, {
                "NormalizeImage": {
                    "scale": 0.00392157,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "order": ''
                }
            }, {
                "ToCHWImage": None
            }]
        },
        "PostProcess": {
            "main_indicator": "Topk",
            "Topk": {
                "topk": topk,
                "class_id_map_file": imagenet1k_map_path
            }
        }
    }
    if "save_dir" in kwargs:
        if kwargs["save_dir"] is not None:
            cfg["PostProcess"]["SavePreLabel"] = {
                "save_dir": kwargs["save_dir"]
            }
    if "class_id_map_file" in kwargs:
        if kwargs["class_id_map_file"] is not None:
            cfg["PostProcess"]["Topk"]["class_id_map_file"] = kwargs[
                "class_id_map_file"]

    cfg = config.AttrDict(cfg)
    config.create_attr_dict(cfg)
    return cfg


def args_cfg():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer_imgs",
        type=str,
        required=True,
        help="The image(s) to be predicted.")
    parser.add_argument(
        "--model_name", type=str, help="The model name to be used.")
    parser.add_argument(
        "--inference_model_dir",
        type=str,
        help="The directory of model files. Valid when model_name not specifed."
    )
    parser.add_argument(
        "--use_gpu", type=str, default=True, help="Whether use GPU.")
    parser.add_argument("--gpu_mem", type=int, default=8000, help="")
    parser.add_argument(
        "--enable_mkldnn",
        type=str2bool,
        default=False,
        help="Whether use MKLDNN. Valid when use_gpu is False")
    parser.add_argument("--cpu_num_threads", type=int, default=1, help="")
    parser.add_argument(
        "--use_tensorrt", type=str2bool, default=False, help="")
    parser.add_argument("--use_fp16", type=str2bool, default=False, help="")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size. Default by 1.")
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Return topk score(s) and corresponding results. Default by 5.")
    parser.add_argument(
        "--class_id_map_file",
        type=str,
        help="The path of file that map class_id and label.")
    parser.add_argument(
        "--save_dir",
        type=str,
        help="The directory to save prediction results as pre-label.")
    parser.add_argument(
        "--resize_short",
        type=int,
        default=256,
        help="Resize according to short size.")
    parser.add_argument(
        "--crop_size", type=int, default=224, help="Centor crop size.")

    args = parser.parse_args()
    return vars(args)


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
            f"Something went wrong while downloading file from {url}")


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


class PaddleClas(object):
    """PaddleClas.
    """

    print_info()

    def __init__(self,
                 model_name: str=None,
                 inference_model_dir: str=None,
                 use_gpu: bool=True,
                 batch_size: int=1,
                 topk: int=5,
                 **kwargs):
        """Init PaddleClas with config.

        Args:
            model_name (str, optional): The model name supported by PaddleClas. If specified, override config. Defaults to None.
            inference_model_dir (str, optional): The directory that contained model file and params file to be used. If specified, override config. Defaults to None.
            use_gpu (bool, optional): Whether use GPU. If specified, override config. Defaults to True.
            batch_size (int, optional): The batch size to pridict. If specified, override config. Defaults to 1.
            topk (int, optional): Return the top k prediction results with the highest score. Defaults to 5.
        """
        super().__init__()
        self._config = init_config(model_name, inference_model_dir, use_gpu,
                                   batch_size, topk, **kwargs)
        self._check_input_model()
        self.cls_predictor = ClsPredictor(self._config)

    def get_config(self):
        """Get the config.
        """
        return self._config

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
            if input_model_name not in candidate_model_names:
                err = f"{input_model_name} is not provided by PaddleClas. \nMaybe you want: [{similar_names_str}]. \nIf you want to use your own model, please specify inference_model_dir!"
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
            err = f"Please specify the model name supported by PaddleClas or directory contained model files(inference.pdmodel, inference.pdiparams)."
            raise InputModelError(err)
        return

    def predict(self, input_data: Union[str, np.array],
                print_pred: bool=False) -> Generator[list, None, None]:
        """Predict input_data.

        Args:
            input_data (Union[str, np.array]): 
                When the type is str, it is the path of image, or the directory containing images, or the URL of image from Internet.
                When the type is np.array, it is the image data whose channel order is RGB.
            print_pred (bool, optional): Whether print the prediction result. Defaults to False. Defaults to False.

        Raises:
            ImageTypeError: Illegal input_data.

        Yields:
            Generator[list, None, None]: 
                The prediction result(s) of input_data by batch_size. For every one image, 
                prediction result(s) is zipped as a dict, that includs topk "class_ids", "scores" and "label_names". 
                The format is as follow: [{"class_ids": [...], "scores": [...], "label_names": [...]}, ...]
        """

        if isinstance(input_data, np.ndarray):
            outputs = self.cls_predictor.predict(input_data)
            yield self.cls_predictor.postprocess(outputs)
        elif isinstance(input_data, str):
            if input_data.startswith("http") or input_data.startswith("https"):
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
            topk = self._config.PostProcess.get('topk', 1)

            img_list = []
            img_path_list = []
            cnt = 0
            for idx, img_path in enumerate(image_list):
                img = cv2.imread(img_path)
                if img is None:
                    warnings.warn(
                        f"Image file failed to read and has been skipped. The path: {img_path}"
                    )
                    continue
                img = img[:, :, ::-1]
                img_list.append(img)
                img_path_list.append(img_path)
                cnt += 1

                if cnt % batch_size == 0 or (idx + 1) == len(image_list):
                    outputs = self.cls_predictor.predict(img_list)
                    preds = self.cls_predictor.postprocess(outputs,
                                                           img_path_list)
                    if print_pred and preds:
                        for pred in preds:
                            filename = pred.pop("file_name")
                            pred_str = ", ".join(
                                [f"{k}: {pred[k]}" for k in pred])
                            print(
                                f"filename: {filename}, top-{topk}, {pred_str}")

                    img_list = []
                    img_path_list = []
                    yield preds
        else:
            err = "Please input legal image! The type of image supported by PaddleClas are: NumPy.ndarray and string of local path or Ineternet URL"
            raise ImageTypeError(err)
        return


# for CLI
def main():
    """Function API used for commad line.
    """
    cfg = args_cfg()
    clas_engine = PaddleClas(**cfg)
    res = clas_engine.predict(cfg["infer_imgs"], print_pred=True)
    for _ in res:
        pass
    print("Predict complete!")
    return


if __name__ == "__main__":
    main()
