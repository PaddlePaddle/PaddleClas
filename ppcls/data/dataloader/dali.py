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
from typing import Any, Callable, Dict, List, Tuple, Union
from collections import defaultdict

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.pipeline as pipeline
import nvidia.dali.types as types
import paddle

from nvidia.dali.plugin.paddle import DALIGenericIterator
import nvidia.dali.fn as fn

from dali_operators import DecodeImage
from dali_operators import ResizeImage
from dali_operators import CropImage
from dali_operators import RandomCropImage
from dali_operators import RandCropImage
from dali_operators import RandCropImageV2
from dali_operators import RandFlipImage
from dali_operators import NormalizeImage
from dali_operators import ToCHWImage
from dali_operators import ColorJitter
from dali_operators import RandomRotation
from dali_operators import Pad
from dali_operators import RandomRot90
from dali_operators import DecodeRandomResizedCrop
from dali_operators import CropMirrorNormalize

INTERP_MAP = {
    "nearest": types.DALIInterpType.INTERP_NN,  # cv2.INTER_NEAREST
    "bilinear": types.DALIInterpType.INTERP_LINEAR,  # cv2.INTER_LINEAR
    "bicubic": types.DALIInterpType.INTERP_CUBIC,  # cv2.INTER_CUBIC
    "lanczos": types.DALIInterpType.INTERP_LANCZOS3,  # cv2.INTER_LANCZOS4
}


def make_pair(x: Union[Any, Tuple[Any], List[Any]]) -> Tuple[Any]:
    """repeat input x to be an tuple

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
        op_name (str): preprocess OP name

    Returns:
        Dict[str, Any]: converted params for DALI initialization
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
        if device is not None:
            dali_op_cfg.update({"device": device})
        dali_op_cfg.update({
            "output_type": types.DALIImageType.RGB
            if to_rgb else types.DALIImageType.BGR
        })
        dali_op_cfg.update({
            "device_memory_padding": op_cfg.get("device_memory_padding",
                                                211025920)
        })
        dali_op_cfg.update({
            "host_memory_padding": op_cfg.get("host_memory_padding", 140544512)
        })
    elif op_name == "ResizeImage":
        size = op_cfg.get("size", None)
        size = make_pair(size)
        resize_short = op_cfg.get("resize_short", None)
        interpolation = op_cfg.get("interpolation", None)
        if size is not None:
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
        size = op_cfg.get("size", None)
        size = make_pair(size)
        if size is not None:
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
        size = op_cfg.get("size", None)
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
        # each value of mean should be rescaled in [0, 1/scale] according to
        # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops_legacy.html?highlight=cropmirrornormalize#nvidia.dali.ops.Normalize
        mean = [v / scale for v in mean]
        std = op_cfg.get("std", [0.229, 0.224, 0.225])
        output_fp16 = op_cfg.get("output_fp16", False)
        dali_op_cfg.update({"scale": scale})
        dali_op_cfg.update({
            "mean": np.reshape(
                np.array(
                    mean, dtype="float32"), [1, 1, 3])
        })
        dali_op_cfg.update({
            "stddev": np.reshape(
                np.array(
                    std, dtype="float32"), [1, 1, 3])
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
        interpolation = op_cfg.get("interpolation", "nearest")
        dali_op_cfg.update({"prob": prob})
        dali_op_cfg.update({"angle": degrees})
        dali_op_cfg.update({"interp_type": INTERP_MAP[interpolation]})
    elif op_name == "Pad":
        size = op_cfg.get("size", None)
        assert size is not None, f"`size` can't be None when using DALI, but got {size}"
        make_pair(size)
        padding = op_cfg.get("padding", 0)
        fill = op_cfg.get("fill", 0)
        dali_op_cfg.update({
            "crop_h": size[1] + padding,
            "crop_w": size[0] + padding
        })
        dali_op_cfg.update({"fill_values": fill})
        dali_op_cfg.update({"out_of_bounds_policy": "pad"})
    elif op_name == "RandomRot90":
        interpolation = op_cfg.get("interpolation", "nearest")
        dali_op_cfg.update({"interp_type": INTERP_MAP[interpolation]})
    elif op_name == "DecodeRandomResizedCrop":
        device = "cpu" if device == "cpu" else "mixed"
        output_type = op_cfg.get("output_type", types.DALIImageType.RGB)
        device_memory_padding = op_cfg.get("device_memory_padding", 211025920)
        host_memory_padding = op_cfg.get("host_memory_padding", 140544512)
        scale = op_cfg.get("scale", [0.08, 1.0])
        ratio = op_cfg.get("ratio", [3.0 / 4, 4.0 / 3])
        num_attempts = op_cfg.get("num_attempts", 100)
        if device is not None:
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
    elif op_name == "CropMirrorNormalize":
        dtype = types.FLOAT16 if op_cfg.get("output_fp16",
                                            False) else types.FLOAT
        output_layout = op_cfg.get("output_layout", "CHW")
        crop = op_cfg.get("crop", None)
        scale = op_cfg.get("scale", 1 / 255.0)
        if isinstance(scale, str):
            scale = eval(scale)
        mean = op_cfg.get("mean", [0.485, 0.456, 0.406])
        # each value of mean should be rescaled in [0, 1/scale] according to
        # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops_legacy.html?highlight=cropmirrornormalize#nvidia.dali.ops.CropMirrorNormalize
        mean = [v / scale for v in mean]
        std = op_cfg.get("std", [0.229, 0.224, 0.225])
        pad_output = op_cfg.get("channel_num", 3) == 4
        if dtype is not None:
            dali_op_cfg.update({"dtype": dtype})
        if output_layout is not None:
            dali_op_cfg.update({"output_layout": output_layout})
        if crop is not None:
            dali_op_cfg.update({"crop": crop})
        if scale is not None:
            dali_op_cfg.update({"scale": scale})
        if mean is not None:
            dali_op_cfg.update({"mean": mean})
        if std is not None:
            dali_op_cfg.update({"std": std})
        if pad_output is not None:
            dali_op_cfg.update({"pad_output": pad_output})
    else:
        raise ValueError(f"Operator '{op_name}' is not implemented now.")
    if "device" not in dali_op_cfg:
        dali_op_cfg.update({"device": device})
    return dali_op_cfg


def build_dali_transforms(op_cfg_list: List[Dict[str, Any]],
                          device: str="cpu",
                          fuse: bool=True) -> List[Callable]:
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
        device (str): device which dali operator(s) applied in. Defaults to "cpu".
        fuse (bool): whether to use fused dali operators instead of single operators. Defaults to True.
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
        fused_flag = False
        if fuse:
            # fuse operator if enabled
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
                    fused_flag = True
                    print(
                        f"DALI Operator conversion: [DecodeImage, RandCropImage] -> {dali_op_list[-1].__class__.__name__}"
                    )
            elif 0 < idx and idx + 1 < num_cfg_node:
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
                    fused_flag = True
                    print(
                        f"DALI Operator conversion: [RandCropImage, RandFlipImage, NormalizeImage] -> {dali_op_list[-1].__class__.__name__}"
                    )
        if not fuse or not fused_flag:
            assert isinstance(op_cfg,
                              dict) and len(op_cfg) == 1, "yaml format error"
            if op_name == "Pad":
                op_param.update({
                    "size": parse_value_with_key(op_cfg_list, "size")
                })
            dali_param = convert_cfg_to_dali(op_name, device, **op_param)
            dali_op = eval(op_name)(**dali_param)
            dali_op_list.append(dali_op)
            idx += 1
            print(
                f"DALI Operator conversion: {op_name} -> {dali_op_list[-1].__class__.__name__}"
            )
    return dali_op_list


class HybridPipeline(pipeline.Pipeline):
    def __init__(self,
                 device: str,
                 batch_size: int,
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
            batch_size, num_threads, device_id, seed=seed)
        self.device = device
        if ext_src:
            self.ext_src = True
            self.ext_src = ext_src
        else:
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
                dtype=[types.DALIDataType.UINT8, types.DALIDataType.INT64])
        else:
            raw_images, labels = self.reader(name="Reader")
        images = self.transforms(raw_images)
        return [
            images, self.cast(labels.gpu() if self.device == "gpu" else labels)
        ]

    def __len__(self):
        return self.epoch_size("Reader")


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


def dali_dataloader(config, mode, device, num_threads=4, seed=None):
    assert "gpu" in device, "gpu training is required for DALI"
    device_id = int(device.split(":")[1])
    device = "gpu"
    config_dataloader = config[mode]
    seed = 42 if seed is None else seed
    env = os.environ
    num_gpus = paddle.distributed.get_world_size()

    batch_size = config_dataloader["sampler"]["batch_size"]
    file_root = config_dataloader["dataset"]["image_root"]
    file_list = config_dataloader["dataset"]["cls_label_path"]
    sampler_name = config_dataloader["sampler"].get("name",
                                                    "DistributedBatchSampler")
    dali_transforms = build_dali_transforms(
        config_dataloader["dataset"]["transform_ops"], device)

    if mode.lower() == "train":
        if sampler_name in ["PKSampler", "DistributedRandomIdentitySampler"]:
            ext_src = ExternalRandomIdentityIterator(
                batch_size,
                config_dataloader["dataset"]["sample_per_id" if sampler_name ==
                                             "PKSampler" else "num_instances"],
                device_id,
                num_gpus,
                file_root,
                file_list,
                None,
                relabel=config_dataloader["dataset"].get("relabel", False),
                sample_method=config_dataloader["sampler"].get("sample_method",
                                                               "id_avg_prob"),
                drop_last=config_dataloader["sampler"].get("drop_last", True),
                seed=seed, )
        else:
            ext_src = None
        if "PADDLE_TRAINER_ID" in env and "PADDLE_TRAINERS_NUM" in env and "FLAGS_selected_gpus" in env:
            shard_id = int(env["PADDLE_TRAINER_ID"])
            num_shards = int(env["PADDLE_TRAINERS_NUM"])
            device_id = int(env["FLAGS_selected_gpus"])
            pipe = HybridPipeline(device, batch_size, num_threads, device_id,
                                  seed + shard_id, file_root, file_list,
                                  dali_transforms, shard_id, num_shards, True,
                                  ext_src)
            #  sample_per_shard = len(pipe) // num_shards
        else:
            pipe = HybridPipeline(device, batch_size, 1, device_id, seed,
                                  file_root, file_list, dali_transforms, 0, 1,
                                  False, ext_src)
            #  sample_per_shard = len(pipelines[0])
        pipe.build()
        pipelines = [pipe]
        return DALIImageNetIterator(
            pipelines, ["data", "label"], reader_name="Reader")
    else:
        assert sampler_name in ["DistributedBatchSampler"], \
            f"sampler_name({sampler_name}) must in [\"DistributedBatchSampler\"]"
        if "PADDLE_TRAINER_ID" in env and "PADDLE_TRAINERS_NUM" in env and "FLAGS_selected_gpus" in env and sampler_name == "DistributedBatchSampler":
            shard_id = int(env["PADDLE_TRAINER_ID"])
            num_shards = int(env["PADDLE_TRAINERS_NUM"])
            device_id = int(env["FLAGS_selected_gpus"])

            pipe = HybridPipeline(device, batch_size, num_threads, device_id,
                                  seed, file_root, file_list, dali_transforms,
                                  shard_id, num_shards, False)
        else:
            pipe = HybridPipeline(device, batch_size, 1, device_id, seed,
                                  file_root, file_list, dali_transforms, 0, 1,
                                  False)
        pipe.build()
        pipelines = [pipe]
        return DALIImageNetIterator(
            pipelines, ["data", "label"], reader_name="Reader")


class ExternalRandomIdentityIterator(object):
    def __init__(self,
                 batch_size,
                 num_instances,
                 device_id,
                 num_gpus,
                 image_root,
                 cls_label_path,
                 delimiter=None,
                 relabel=False,
                 sample_method="sample_avg_prob",
                 drop_last=True,
                 seed=None):
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.device_id = device_id
        self._img_root = image_root
        self._cls_path = cls_label_path
        self.delimiter = delimiter if delimiter is not None else " "
        self.relabel = relabel
        self.seed = seed
        self.sample_method = sample_method
        self.images = []
        self.labels = []
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

            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for line in lines:
                line = line.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, line[0]))
                if self.relabel:
                    self.labels.append(label_map[np.int64(line[1])])
                else:
                    self.labels.append(np.int64(line[1]))
                assert os.path.exists(self.images[
                    -1]), f"path {self.images[-1]} does not exist."
        # whole data set size
        self.data_set_len = len(self.images)
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.n = self.data_set_len // num_gpus
        self.drop_last = drop_last

    def __iter__(self):
        self.i = 0
        tmp = list(zip(self.images, self.labels))
        if self.seed is not None:
            np.random.RandomState(self.seed + self.device_id).shuffle(tmp)
        self.images, self.labels = zip(*tmp)

        self.label_dict = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_dict[label].append(idx)
        self.label_list = list(self.label_dict)
        if self.sample_method == "id_avg_prob":
            self.prob_list = np.array([1 / len(self.label_list)] *
                                      len(self.label_list))
        elif self.sample_method == "sample_avg_prob":
            counter = []
            for label_i in self.label_list:
                counter.append(len(self.label_dict[label_i]))
            self.prob_list = np.array(counter) / sum(counter)

        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."
        return self

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        for _ in range(len(self)):
            batch_index = []
            batch_label_list = np.random.choice(
                self.label_list,
                size=self.num_pids_per_batch,
                replace=False,
                p=self.prob_list)
            for label_i in batch_label_list:
                label_i_indexes = self.label_dict[label_i]
                if self.num_instances <= len(label_i_indexes):
                    batch_index.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.num_instances,
                            replace=False))
                else:
                    batch_index.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.num_instances,
                            replace=True))
            if not self.drop_last or len(batch_index) == self.batch_size:
                break
        batch = []
        labels = []
        for index in batch_index:
            batch.append(np.fromfile(self.images[index], dtype="uint8"))
            labels.append(np.int64(self.labels[index]))
            self.i += 1
        return (batch, labels)

    def __len__(self):
        return self.data_set_len


if __name__ == "__main__":
    shituv2_config = {
        "dataset": {
            "name": "ImageNetDataset",
            "image_root": "./dataset/",
            "cls_label_path": "./dataset/train_reg_all_data_v2.txt",
            "relabel": True,
            "transform_ops": [{
                "DecodeImage": {
                    "to_rgb": True,
                    "channel_first": False
                }
            }, {
                "ResizeImage": {
                    "size": [224, 224],
                    "return_numpy": False,
                    "interpolation": "bilinear",
                    "backend": "cv2"
                }
            }, {
                "RandFlipImage": {
                    "flip_code": 1
                }
            }, {
                "Pad": {
                    "padding": 10,
                    "backend": "cv2"
                }
            }, {
                "RandCropImageV2": {
                    "size": [224, 224]
                }
            }, {
                "RandomRotation": {
                    "prob": 0.5,
                    "degrees": 90,
                    "interpolation": "bilinear"
                }
            }, {
                "ResizeImage": {
                    "size": [224, 224],
                    "return_numpy": False,
                    "interpolation": "bilinear",
                    "backend": "cv2"
                }
            }, {
                "NormalizeImage": {
                    "scale": "1.0/255.0",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "order": "hwc"
                }
            }]
        }
    }
    shituv1_config = {
        "dataset": {
            "name": "ImageNetDataset",
            "image_root": "./dataset/",
            "cls_label_path": "./dataset/train_reg_all_data.txt",
            "transform_ops": [{
                "DecodeImage": {
                    "to_rgb": True,
                    "channel_first": False
                }
            }, {
                "RandCropImage": {
                    "size": 224
                }
            }, {
                "RandFlipImage": {
                    "flip_code": 1
                }
            }, {
                "NormalizeImage": {
                    "scale": "1.0/255.0",
                    "mean": [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
                    "std": [0.229, 0.224, 0.225],
                    "order": ""
                }
            }]
        }
    }

    import cv2
    device = "gpu"
    dali_transforms = build_dali_transforms(
        shituv2_config["dataset"]["transform_ops"], device)
    batch_size = 4
    num_threads = 4
    seed = 42
    # file_root = "/workspace/hesensen/dali_learning/DALI/docs/examples/data/images"
    file_root = "/workspace/hesensen/PaddleClas_DALI/dataset/ppshitu_data/"
    file_list = "/workspace/hesensen/PaddleClas_DALI/dataset/ppshitu_data/debug.txt"
    ext_rdn_id_iter = ExternalRandomIdentityIterator(
        batch_size,
        2,
        0,
        1,
        file_root,
        file_list,
        relabel=True,
        sample_method="id_avg_prob",
        seed=seed)
    pipe = HybridPipeline(
        device,
        batch_size,
        num_threads,
        0,
        seed,
        file_root,
        file_list,
        dali_transforms,
        ext_src=ext_rdn_id_iter)
    pipe.build()
    pipelines = [pipe]
    dali_loader = DALIImageNetIterator(pipelines, ["data", "label"])

    for iter_id, batch in enumerate(dali_loader):
        images, labels = batch
        print(images.place, images.shape)
        print(labels.place, labels.shape)
        print(labels)
        img_list = []
        for j in range(batch_size):
            img = images.numpy()[j]
            if img.shape[0] == 3:
                img = img.transpose((1, 2, 0))
            img_list.append(img)
        img_list = np.concatenate(img_list, axis=0)
        img_list = ((
            img_list * np.reshape(np.array([0.229, 0.224, 0.225]), [1, 1, 3]) +
            np.reshape(np.array([0.485, 0.456, 0.406]), [1, 1, 3])))
        img_list = (img_list * 255.0).clip(0, 255.0).astype("uint8")
        cv2.imwrite(f"./pipeline_output_{iter_id}.jpg", img_list[:, :, ::-1])
