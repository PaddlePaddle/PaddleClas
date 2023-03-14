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

from __future__ import division

import copy
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.pipeline as pipeline
import nvidia.dali.types as types
import paddle
from nvidia.dali.plugin.paddle import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from ppcls.data.preprocess.ops.dali_operators import ColorJitter
from ppcls.data.preprocess.ops.dali_operators import CropImage
from ppcls.data.preprocess.ops.dali_operators import CropMirrorNormalize
from ppcls.data.preprocess.ops.dali_operators import DecodeImage
from ppcls.data.preprocess.ops.dali_operators import DecodeRandomResizedCrop
from ppcls.data.preprocess.ops.dali_operators import NormalizeImage
from ppcls.data.preprocess.ops.dali_operators import Pad
from ppcls.data.preprocess.ops.dali_operators import RandCropImage
from ppcls.data.preprocess.ops.dali_operators import RandCropImageV2
from ppcls.data.preprocess.ops.dali_operators import RandFlipImage
from ppcls.data.preprocess.ops.dali_operators import RandomCropImage
from ppcls.data.preprocess.ops.dali_operators import RandomRot90
from ppcls.data.preprocess.ops.dali_operators import RandomRotation
from ppcls.data.preprocess.ops.dali_operators import ResizeImage
from ppcls.data.preprocess.ops.dali_operators import ToCHWImage
from ppcls.engine.train.utils import type_name
from ppcls.utils import logger

INTERP_MAP = {
    "nearest": types.DALIInterpType.INTERP_NN,  # cv2.INTER_NEAREST
    "bilinear": types.DALIInterpType.INTERP_LINEAR,  # cv2.INTER_LINEAR
    "bicubic": types.DALIInterpType.INTERP_CUBIC,  # cv2.INTER_CUBIC
    "lanczos": types.DALIInterpType.INTERP_LANCZOS3,  # cv2.INTER_LANCZOS4
}


def make_pair(x: Union[Any, Tuple[Any], List[Any]]) -> Tuple[Any]:
    """repeat input x to be an tuple if x is an single element, else return x directly

    Args:
        x (Union[Any, Tuple[Any], List[Any]]): input x

    Returns:
        Tuple[Any]: tupled input
    """
    return x if isinstance(x, (tuple, list)) else (x, x)


def parse_value_with_key(content: Union[Dict, List[Dict]],
                         key: str) -> Union[None, Any]:
    """parse value according to given key recursively, return None if not found

    Args:
        content (Union[Dict, List[Dict]]): content to be parsed
        key (str): given key

    Returns:
        Union[None, Any]: result
    """
    if isinstance(content, dict):
        if key in content:
            return content[key]
        for content_ in content.values():
            value = parse_value_with_key(content_, key)
            if value is not None:
                return value
    elif isinstance(content, (tuple, list)):
        for content_ in content:
            value = parse_value_with_key(content_, key)
            if value is not None:
                return value
    return None


def convert_cfg_to_dali(op_name: str, device: str, **op_cfg) -> Dict[str, Any]:
    """convert original preprocess op params into DALI-based op params

    Args:
        op_name (str): name of operator
        device (str): device which operator applied on

    Returns:
        Dict[str, Any]: converted arguments for DALI initialization
    """
    assert device in ["cpu", "gpu"
                      ], f"device({device}) must in [\"cpu\", \"gpu\"]"
    dali_op_cfg = {}
    if op_name == "DecodeImage":
        device = "cpu" if device == "cpu" else "mixed"
        to_rgb = op_cfg.get("to_rgb", True)
        channel_first = op_cfg.get("channel_first", False)
        assert channel_first is False, \
            f"`channel_first` must set to False when using DALI, but got {channel_first}"
        dali_op_cfg.update({"device": device})
        dali_op_cfg.update({
            "output_type": types.DALIImageType.RGB
            if to_rgb else types.DALIImageType.BGR
        })
        dali_op_cfg.update({
            "device_memory_padding":
            op_cfg.get("device_memory_padding", 211025920)
        })
        dali_op_cfg.update({
            "host_memory_padding": op_cfg.get("host_memory_padding", 140544512)
        })
    elif op_name == "ResizeImage":
        size = op_cfg.get("size", None)
        resize_short = op_cfg.get("resize_short", None)
        interpolation = op_cfg.get("interpolation", None)
        if size is not None:
            size = make_pair(size)
            dali_op_cfg.update({"resize_y": size[0], "resize_x": size[1]})
        if resize_short is not None:
            dali_op_cfg.update({"resize_shorter": resize_short})
        if interpolation is not None:
            dali_op_cfg.update({"interp_type": INTERP_MAP[interpolation]})
    elif op_name == "CropImage":
        size = op_cfg.get("size", 224)
        size = make_pair(size)
        dali_op_cfg.update({"crop_h": size[1], "crop_w": size[0]})
        dali_op_cfg.update({"crop_pos_x": 0.5, "crop_pos_y": 0.5})
    elif op_name == "RandomCropImage":
        size = op_cfg.get("size", 224)
        if size is not None:
            size = make_pair(size)
            dali_op_cfg.update({"crop_h": size[1], "crop_w": size[0]})
    elif op_name == "RandCropImage":
        size = op_cfg.get("size", 224)
        size = make_pair(size)
        scale = op_cfg.get("scale", [0.08, 1.0])
        ratio = op_cfg.get("ratio", [3.0 / 4, 4.0 / 3])
        interpolation = op_cfg.get("interpolation", "bilinear")
        dali_op_cfg.update({"size": size})
        if scale is not None:
            dali_op_cfg.update({"random_area": scale})
        if ratio is not None:
            dali_op_cfg.update({"random_aspect_ratio": ratio})
        if interpolation is not None:
            dali_op_cfg.update({"interp_type": INTERP_MAP[interpolation]})
    elif op_name == "RandCropImageV2":
        size = op_cfg.get("size", 224)
        size = make_pair(size)
        dali_op_cfg.update({"crop_h": size[1], "crop_w": size[0]})
    elif op_name == "RandFlipImage":
        prob = op_cfg.get("prob", 0.5)
        flip_code = op_cfg.get("flip_code", 1)
        dali_op_cfg.update({"prob": prob})
        dali_op_cfg.update({"flip_code": flip_code})
    elif op_name == "NormalizeImage":
        # scale * (in - mean) / stddev + shift
        scale = op_cfg.get("scale", 1.0 / 255.0)
        if isinstance(scale, str):
            scale = eval(scale)
        mean = op_cfg.get("mean", [0.485, 0.456, 0.406])
        std = op_cfg.get("std", [0.229, 0.224, 0.225])
        mean = [v / scale for v in mean]
        std = [v / scale for v in std]
        order = op_cfg.get("order", "chw")
        channel_num = op_cfg.get("channel_num", 3)
        output_fp16 = op_cfg.get("output_fp16", False)
        dali_op_cfg.update({
            "mean": np.reshape(
                np.array(
                    mean, dtype="float32"), [channel_num, 1, 1]
                if order == "chw" else [1, 1, channel_num])
        })
        dali_op_cfg.update({
            "stddev": np.reshape(
                np.array(
                    std, dtype="float32"), [channel_num, 1, 1]
                if order == "chw" else [1, 1, channel_num])
        })
        if output_fp16:
            dali_op_cfg.update({"dtype": types.FLOAT16})
    elif op_name == "ToCHWImage":
        dali_op_cfg.update({"perm": [2, 0, 1]})
    elif op_name == "ColorJitter":
        prob = op_cfg.get("prob", 1.0)
        brightness = op_cfg.get("brightness", 0.0)
        contrast = op_cfg.get("contrast", 0.0)
        saturation = op_cfg.get("saturation", 0.0)
        hue = op_cfg.get("hue", 0.0)
        dali_op_cfg.update({"prob": prob})
        dali_op_cfg.update({"brightness_factor": brightness})
        dali_op_cfg.update({"contrast_factor": contrast})
        dali_op_cfg.update({"saturation_factor": saturation})
        dali_op_cfg.update({"hue_factor": hue})
    elif op_name == "RandomRotation":
        prob = op_cfg.get("prob", 0.5)
        degrees = op_cfg.get("degrees", 90)
        interpolation = op_cfg.get("interpolation", "bilinear")
        dali_op_cfg.update({"prob": prob})
        dali_op_cfg.update({"angle": degrees})
        dali_op_cfg.update({"interp_type": INTERP_MAP[interpolation]})
    elif op_name == "Pad":
        size = op_cfg.get("size", 224)
        size = make_pair(size)
        padding = op_cfg.get("padding", 0)
        fill = op_cfg.get("fill", 0)
        dali_op_cfg.update({
            "crop_h": padding + size[1] + padding,
            "crop_w": padding + size[0] + padding
        })
        dali_op_cfg.update({"fill_values": fill})
        dali_op_cfg.update({"out_of_bounds_policy": "pad"})
    elif op_name == "RandomRot90":
        interpolation = op_cfg.get("interpolation", "nearest")
    elif op_name == "DecodeRandomResizedCrop":
        device = "cpu" if device == "cpu" else "mixed"
        output_type = op_cfg.get("output_type", types.DALIImageType.RGB)
        device_memory_padding = op_cfg.get("device_memory_padding", 211025920)
        host_memory_padding = op_cfg.get("host_memory_padding", 140544512)
        scale = op_cfg.get("scale", [0.08, 1.0])
        ratio = op_cfg.get("ratio", [3.0 / 4, 4.0 / 3])
        num_attempts = op_cfg.get("num_attempts", 100)
        size = op_cfg.get("size", 224)
        dali_op_cfg.update({"device": device})
        if output_type is not None:
            dali_op_cfg.update({"output_type": output_type})
        if device_memory_padding is not None:
            dali_op_cfg.update({
                "device_memory_padding": device_memory_padding
            })
        if host_memory_padding is not None:
            dali_op_cfg.update({"host_memory_padding": host_memory_padding})
        if scale is not None:
            dali_op_cfg.update({"random_area": scale})
        if ratio is not None:
            dali_op_cfg.update({"random_aspect_ratio": ratio})
        if num_attempts is not None:
            dali_op_cfg.update({"num_attempts": num_attempts})
        if size is not None:
            dali_op_cfg.update({"resize_x": size, "resize_y": size})
    elif op_name == "CropMirrorNormalize":
        dtype = types.FLOAT16 if op_cfg.get("output_fp16",
                                            False) else types.FLOAT
        output_layout = op_cfg.get("output_layout", "CHW")
        size = op_cfg.get("size", None)
        scale = op_cfg.get("scale", 1 / 255.0)
        if isinstance(scale, str):
            scale = eval(scale)
        mean = op_cfg.get("mean", [0.485, 0.456, 0.406])
        mean = [v / scale for v in mean]
        std = op_cfg.get("std", [0.229, 0.224, 0.225])
        std = [v / scale for v in std]
        pad_output = op_cfg.get("channel_num", 3) == 4
        prob = op_cfg.get("prob", 0.5)
        dali_op_cfg.update({"dtype": dtype})
        if output_layout is not None:
            dali_op_cfg.update({"output_layout": output_layout})
        if size is not None:
            dali_op_cfg.update({"crop": (size, size)})
        if mean is not None:
            dali_op_cfg.update({"mean": mean})
        if std is not None:
            dali_op_cfg.update({"std": std})
        if pad_output is not None:
            dali_op_cfg.update({"pad_output": pad_output})
        if prob is not None:
            dali_op_cfg.update({"prob": prob})
    else:
        raise ValueError(
            f"DALI operator \"{op_name}\"  in PaddleClas is not implemented now. please refer to docs/zh_CN/training/config_description/develop_with_DALI.md"
        )
    if "device" not in dali_op_cfg:
        dali_op_cfg.update({"device": device})
    return dali_op_cfg


def build_dali_transforms(op_cfg_list: List[Dict[str, Any]],
                          mode: str,
                          device: str="gpu",
                          enable_fuse: bool=True) -> List[Callable]:
    """create dali operators based on the config
    Args:
        op_cfg_list (List[Dict[str, Any]]): a dict list, used to create some operators, such as config below
            --------------------------------
            - DecodeImage:
                to_rgb: True
                channel_first: False
            - ResizeImage:
                size: 224
            - NormalizeImage:
                scale: 0.00392157
                mean: [0.485, 0.456, 0.406]
                std: [0.229, 0.224, 0.225]
                order: ""
            --------------------------------
        mode (str): mode.
        device (str): device which dali operator(s) applied in. Defaults to "gpu".
        enable_fuse (bool): whether to use fused dali operators instead of single operators, such as DecodeRandomResizedCrop. Defaults to True.
    Returns:
        List[Callable]: Callable DALI operators in list.
    """
    assert isinstance(op_cfg_list, list), "operator config should be a list"
    # build dali transforms list

    dali_op_list = []
    idx = 0
    num_cfg_node = len(op_cfg_list)
    while idx < num_cfg_node:
        op_cfg = op_cfg_list[idx]
        op_name = list(op_cfg)[0]
        op_param = {} if op_cfg[op_name] is None else copy.deepcopy(op_cfg[
            op_name])
        fused_success = False
        if enable_fuse:
            # fuse operators if enabled
            if idx + 1 < num_cfg_node:
                op_name_nxt = list(op_cfg_list[idx + 1])[0]
                if (op_name == "DecodeImage" and
                        op_name_nxt == "RandCropImage"):
                    fused_op_name = "DecodeRandomResizedCrop"
                    fused_op_param = convert_cfg_to_dali(
                        fused_op_name, device, **{
                            ** op_param, ** (op_cfg_list[idx + 1][op_name_nxt])
                        })
                    fused_dali_op = eval(fused_op_name)(**fused_op_param)
                    idx += 2
                    dali_op_list.append(fused_dali_op)
                    fused_success = True
                    logger.info(
                        f"DALI fused Operator conversion({mode}): [DecodeImage, RandCropImage] -> {type_name(dali_op_list[-1])}: {fused_op_param}"
                    )
            if not fused_success and 0 < idx and idx + 1 < num_cfg_node:
                op_name_pre = list(op_cfg_list[idx - 1])[0]
                op_name_nxt = list(op_cfg_list[idx + 1])[0]
                if (op_name_pre == "RandCropImage" and
                        op_name == "RandFlipImage" and
                        op_name_nxt == "NormalizeImage"):
                    fused_op_name = "CropMirrorNormalize"
                    fused_op_param = convert_cfg_to_dali(
                        fused_op_name, device, **{
                            ** op_param, **
                            (op_cfg_list[idx - 1][op_name_pre]), **
                            (op_cfg_list[idx + 1][op_name_nxt])
                        })
                    fused_dali_op = eval(fused_op_name)(**fused_op_param)
                    idx += 2
                    dali_op_list.append(fused_dali_op)
                    fused_success = True
                    logger.info(
                        f"DALI fused Operator conversion({mode}): [RandCropImage, RandFlipImage, NormalizeImage] -> {type_name(dali_op_list[-1])}: {fused_op_param}"
                    )
            if not fused_success and idx + 1 < num_cfg_node:
                op_name_nxt = list(op_cfg_list[idx + 1])[0]
                if (op_name == "CropImage" and
                        op_name_nxt == "NormalizeImage"):
                    fused_op_name = "CropMirrorNormalize"
                    fused_op_param = convert_cfg_to_dali(
                        fused_op_name, device, **{
                            **
                            op_param,
                            **
                            (op_cfg_list[idx + 1][op_name_nxt]),
                            "prob": 0.0
                        })
                    fused_dali_op = eval(fused_op_name)(**fused_op_param)
                    idx += 2
                    dali_op_list.append(fused_dali_op)
                    fused_success = True
                    logger.info(
                        f"DALI fused Operator conversion({mode}): [CropImage, NormalizeImage] -> {type_name(dali_op_list[-1])}: {fused_op_param}"
                    )
        if not enable_fuse or not fused_success:
            assert isinstance(op_cfg,
                              dict) and len(op_cfg) == 1, "yaml format error"
            if op_name == "Pad":
                # NOTE: Argument `size` must be provided for DALI operator
                op_param.update({
                    "size": parse_value_with_key(op_cfg_list[:idx], "size")
                })
            dali_param = convert_cfg_to_dali(op_name, device, **op_param)
            dali_op = eval(op_name)(**dali_param)
            dali_op_list.append(dali_op)
            idx += 1
            logger.info(
                f"DALI Operator conversion({mode}): {op_name} -> {type_name(dali_op_list[-1])}: {dali_param}"
            )
    return dali_op_list


class ExternalSource_RandomIdentity(object):
    """PKsampler implemented with ExternalSource

    Args:
        batch_size (int): batch size
        sample_per_id (int): number of instance(s) within an class
        device_id (int): device id
        shard_id (int): shard id
        num_gpus (int): number of gpus
        image_root (str): image root directory
        cls_label_path (str): path to annotation file, such as `train_list.txt` or `val_list.txt`
        delimiter (Optional[str], optional): delimiter. Defaults to None.
        relabel (bool, optional): whether do relabel when original label do not starts from 0 or are discontinuous. Defaults to False.
        sample_method (str, optional): sample method when generating prob_list. Defaults to "sample_avg_prob".
        id_list (List[int], optional): list of (start_id, end_id, start_id, end_id) for set of ids to duplicated. Defaults to None.
        ratio (List[Union[int, float]], optional): list of (ratio1, ratio2..) the duplication number for ids in id_list. Defaults to None.
        shuffle (bool): whether to shuffle label list. Defaults to True.
    """

    def __init__(self,
                 batch_size: int,
                 sample_per_id: int,
                 device_id: int,
                 shard_id: int,
                 num_gpus: int,
                 image_root: str,
                 cls_label_path: str,
                 delimiter: Optional[str]=None,
                 relabel: bool=False,
                 sample_method: str="sample_avg_prob",
                 id_list: List[int]=None,
                 ratio: List[Union[int, float]]=None,
                 shuffle: bool=True):
        self.batch_size = batch_size
        self.sample_per_id = sample_per_id
        self.label_per_batch = self.batch_size // self.sample_per_id
        self.device_id = device_id
        self.shard_id = shard_id
        self.num_gpus = num_gpus
        self._img_root = image_root
        self._cls_path = cls_label_path
        self.delimiter = delimiter if delimiter is not None else " "
        self.relabel = relabel
        self.sample_method = sample_method
        self.image_paths = []
        self.labels = []
        self.epoch = 0

        # NOTE: code from ImageNetDataset below
        with open(self._cls_path, "r") as fd:
            lines = fd.readlines()
            if self.relabel:
                label_set = set()
                for line in lines:
                    line = line.strip().split(self.delimiter)
                    label_set.add(np.int64(line[1]))
                label_map = {
                    oldlabel: newlabel
                    for newlabel, oldlabel in enumerate(label_set)
                }

            for line in lines:
                line = line.strip().split(self.delimiter)
                self.image_paths.append(os.path.join(self._img_root, line[0]))
                if self.relabel:
                    self.labels.append(label_map[np.int64(line[1])])
                else:
                    self.labels.append(np.int64(line[1]))
                assert os.path.exists(self.image_paths[
                    -1]), f"path {self.image_paths[-1]} does not exist."

        # NOTE: code from PKSampler below
        # group sample indexes into their label bucket
        self.label_dict = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_dict[label].append(idx)
        # get all label
        self.label_list = list(self.label_dict)
        assert len(self.label_list) * self.sample_per_id >= self.batch_size, \
            f"batch size({self.batch_size}) should not be bigger than than #classes({len(self.label_list)})*sample_per_id({self.sample_per_id})"

        if self.sample_method == "id_avg_prob":
            self.prob_list = np.array([1 / len(self.label_list)] *
                                      len(self.label_list))
        elif self.sample_method == "sample_avg_prob":
            counter = []
            for label_i in self.label_list:
                counter.append(len(self.label_dict[label_i]))
            self.prob_list = np.array(counter) / sum(counter)

        # reweight prob_list according to id_list and ratio if provided
        if id_list and ratio:
            assert len(id_list) % 2 == 0 and len(id_list) == len(ratio) * 2
            for i in range(len(self.prob_list)):
                for j in range(len(ratio)):
                    if i >= id_list[j * 2] and i <= id_list[j * 2 + 1]:
                        self.prob_list[i] = self.prob_list[i] * ratio[j]
                        break
            self.prob_list = self.prob_list / sum(self.prob_list)

        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."

        diff = np.abs(sum(self.prob_list) - 1)
        if diff > 0.00000001:
            self.prob_list[-1] = 1 - sum(self.prob_list[:-1])
            if self.prob_list[-1] > 1 or self.prob_list[-1] < 0:
                logger.error("PKSampler prob list error")
            else:
                logger.info(
                    "sum of prob list not equal to 1, diff is {}, change the last prob".
                    format(diff))

        # whole dataset size
        self.data_set_len = len(self.image_paths)

        # get sharded size
        self.sharded_data_set_len = self.data_set_len // self.num_gpus

        # iteration log
        self.shuffle = shuffle
        self.total_iter = self.sharded_data_set_len // batch_size
        self.iter_count = 0

    def __iter__(self):
        if self.shuffle:
            seed = self.shard_id * 12345 + self.epoch
            np.random.RandomState(seed).shuffle(self.label_list)
            np.random.RandomState(seed).shuffle(self.prob_list)
            self.epoch += 1
        return self

    def __next__(self):
        if self.iter_count >= self.total_iter:
            self.__iter__()
            self.iter_count = 0

        batch_indexes = []
        for _ in range(self.sharded_data_set_len):
            batch_label_list = np.random.choice(
                self.label_list,
                size=self.label_per_batch,
                replace=False,
                p=self.prob_list)
            for label_i in batch_label_list:
                label_i_indexes = self.label_dict[label_i]
                if self.sample_per_id <= len(label_i_indexes):
                    batch_indexes.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.sample_per_id,
                            replace=False))
                else:
                    batch_indexes.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.sample_per_id,
                            replace=True))
            if len(batch_indexes) == self.batch_size:
                break
            batch_indexes = []

        batch_raw_images = []
        batch_labels = []
        for index in batch_indexes:
            batch_raw_images.append(
                np.fromfile(
                    self.image_paths[index], dtype="uint8"))
            batch_labels.append(self.labels[index])

        self.iter_count += 1
        return (batch_raw_images, np.array(batch_labels, dtype="int64"))

    def __len__(self):
        return self.sharded_data_set_len


class HybridPipeline(pipeline.Pipeline):
    """Hybrid Pipeline

    Args:
        device (str): device
        batch_size (int): batch size
        py_num_workers (int): number of python worker(s)
        num_threads (int): number of thread(s)
        device_id (int): device id
        seed (int): random seed
        file_root (str): file root path
        file_list (str): path to annotation file, such as `train_list.txt` or `val_list.txt`
        transform_list (List[Callable]): List of DALI transform operator(s)
        shard_id (int, optional): shard id. Defaults to 0.
        num_shards (int, optional): number of shard(s). Defaults to 1.
        random_shuffle (bool, optional): whether shuffle data during training. Defaults to True.
        ext_src (optional): custom external source. Defaults to None.
    """

    def __init__(self,
                 device: str,
                 batch_size: int,
                 py_num_workers: int,
                 num_threads: int,
                 device_id: int,
                 seed: int,
                 file_root: str,
                 file_list: str,
                 transform_list: List[Callable],
                 shard_id: int=0,
                 num_shards: int=1,
                 random_shuffle: bool=True,
                 ext_src=None):
        super(HybridPipeline, self).__init__(
            batch_size=batch_size,
            device_id=device_id,
            seed=seed,
            py_start_method="fork" if ext_src is None else "spawn",
            py_num_workers=py_num_workers,
            num_threads=num_threads)
        self.device = device
        self.ext_src = ext_src
        if ext_src is None:
            self.reader = ops.readers.File(
                file_root=file_root,
                file_list=file_list,
                shard_id=shard_id,
                num_shards=num_shards,
                random_shuffle=random_shuffle)
        self.transforms = ops.Compose(transform_list)
        self.cast = ops.Cast(dtype=types.DALIDataType.INT64, device=device)

    def define_graph(self):
        if self.ext_src:
            raw_images, labels = fn.external_source(
                source=self.ext_src,
                num_outputs=2,
                dtype=[types.DALIDataType.UINT8, types.DALIDataType.INT64],
                batch=True,
                parallel=True)
        else:
            raw_images, labels = self.reader(name="Reader")
        images = self.transforms(raw_images)
        return [
            images, self.cast(labels.gpu() if self.device == "gpu" else labels)
        ]

    def __len__(self):
        if self.ext_src is not None:
            return len(self.ext_src)
        return self.epoch_size(name="Reader")


class DALIImageNetIterator(DALIGenericIterator):
    def __init__(self, *kargs, **kwargs):
        super(DALIImageNetIterator, self).__init__(*kargs, **kwargs)
        self.in_dynamic_mode = paddle.in_dynamic_mode()

    def __next__(self) -> List[paddle.Tensor]:
        data_batch = super(DALIImageNetIterator,
                           self).__next__()  # List[Dict[str, Tensor], ...]
        # reformat to List[Tensor1, Tensor2, ...]
        data_batch = [
            paddle.to_tensor(data_batch[0][key])
            if self.in_dynamic_mode else data_batch[0][key]
            for key in self.output_map
        ]
        return data_batch


def dali_dataloader(config: Dict[str, Any],
                    mode: str,
                    device: str,
                    py_num_workers: int=1,
                    num_threads: int=4,
                    seed: Optional[int]=None,
                    enable_fuse: bool=True) -> DALIImageNetIterator:
    """build and return HybridPipeline

    Args:
        config (Dict[str, Any]): train/eval dataloader configuration
        mode (str): mode
        device (str): device string
        py_num_workers (int, optional): number of python worker(s). Defaults to 1.
        num_threads (int, optional): number of thread(s). Defaults to 4.
        seed (Optional[int], optional): random seed. Defaults to None.
        enable_fuse (bool, optional): enable fused operator(s). Defaults to True.

    Returns:
        DALIImageNetIterator: Iterable DALI dataloader
    """
    assert "gpu" in device, f"device must be \"gpu\" when running with DALI, but got {device}"
    config_dataloader = config[mode]
    device_id = int(device.split(":")[1])
    device = "gpu"
    seed = 42 if seed is None else seed
    env = os.environ
    num_gpus = paddle.distributed.get_world_size()

    batch_size = config_dataloader["sampler"]["batch_size"]
    file_root = config_dataloader["dataset"]["image_root"]
    file_list = config_dataloader["dataset"]["cls_label_path"]
    sampler_name = config_dataloader["sampler"].get("name",
                                                    "DistributedBatchSampler")
    transform_ops_cfg = config_dataloader["dataset"]["transform_ops"]
    random_shuffle = config_dataloader["sampler"].get("shuffle", None)
    dali_transforms = build_dali_transforms(
        transform_ops_cfg, mode, device, enable_fuse=enable_fuse)
    if "ToCHWImage" not in [type_name(op) for op in dali_transforms] and (
            "CropMirrorNormalize" not in
        [type_name(op) for op in dali_transforms]):
        dali_transforms.append(ToCHWImage(perm=[2, 0, 1], device=device))
        logger.info(
            "Append DALI operator \"ToCHWImage\" at the end of dali_transforms for getting output in \"CHW\" shape"
        )

    if mode.lower() in ["train"]:
        if "PADDLE_TRAINER_ID" in env and "PADDLE_TRAINERS_NUM" in env and "FLAGS_selected_gpus" in env:
            shard_id = int(env["PADDLE_TRAINER_ID"])
            num_shards = int(env["PADDLE_TRAINERS_NUM"])
            device_id = int(env["FLAGS_selected_gpus"])
        else:
            shard_id = 0
            num_shards = 1
        logger.info(
            f"Building DALI {mode} pipeline with num_shards: {num_shards}, num_gpus: {num_gpus}"
        )

        random_shuffle = random_shuffle if random_shuffle is not None else True
        if sampler_name in ["PKSampler", "DistributedRandomIdentitySampler"]:
            ext_src = ExternalSource_RandomIdentity(
                batch_size=batch_size,
                sample_per_id=config_dataloader["sampler"][
                    "sample_per_id"
                    if sampler_name == "PKSampler" else "num_instances"],
                device_id=device_id,
                shard_id=shard_id,
                num_gpus=num_gpus,
                image_root=file_root,
                cls_label_path=file_list,
                delimiter=None,
                relabel=config_dataloader["dataset"].get("relabel", False),
                sample_method=config_dataloader["sampler"].get(
                    "sample_method", "sample_avg_prob"),
                id_list=config_dataloader["sampler"].get("id_list", None),
                ratio=config_dataloader["sampler"].get("ratio", None),
                shuffle=random_shuffle)
            logger.info(
                f"Building DALI {mode} pipeline with ext_src({type_name(ext_src)})"
            )
        else:
            ext_src = None

        pipe = HybridPipeline(device, batch_size, py_num_workers, num_threads,
                              device_id, seed + shard_id, file_root, file_list,
                              dali_transforms, shard_id, num_shards,
                              random_shuffle, ext_src)
        pipe.build()
        pipelines = [pipe]
        if ext_src is None:
            return DALIImageNetIterator(
                pipelines, ["data", "label"], reader_name="Reader")
        else:
            return DALIImageNetIterator(
                pipelines,
                ["data", "label"],
                size=len(ext_src),
                last_batch_policy=LastBatchPolicy.
                DROP  # make reset() successfully
            )
    elif mode.lower() in ["eval", "gallery", "query"]:
        assert sampler_name in ["DistributedBatchSampler"], \
            f"sampler_name({sampler_name}) must in [\"DistributedBatchSampler\"]"
        if "PADDLE_TRAINER_ID" in env and "PADDLE_TRAINERS_NUM" in env and "FLAGS_selected_gpus" in env:
            shard_id = int(env["PADDLE_TRAINER_ID"])
            num_shards = int(env["PADDLE_TRAINERS_NUM"])
            device_id = int(env["FLAGS_selected_gpus"])
        else:
            shard_id = 0
            num_shards = 1
        logger.info(
            f"Building DALI {mode} pipeline with num_shards: {num_shards}, num_gpus: {num_gpus}..."
        )

        random_shuffle = random_shuffle if random_shuffle is not None else False
        pipe = HybridPipeline(device, batch_size, py_num_workers, num_threads,
                              device_id, seed + shard_id, file_root, file_list,
                              dali_transforms, shard_id, num_shards,
                              random_shuffle)
        pipe.build()
        pipelines = [pipe]
        return DALIImageNetIterator(
            pipelines, ["data", "label"], reader_name="Reader")
    else:
        raise ValueError(f"Invalid mode({mode}) when building DALI pipeline")
