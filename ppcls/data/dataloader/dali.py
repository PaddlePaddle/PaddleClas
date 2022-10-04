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

import os
from typing import Any, Callable, Dict, List

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.pipeline as pipeline
import nvidia.dali.types as types
import paddle
from nvidia.dali.plugin.paddle import DALIGenericIterator

# from ppcls.utils import logger


class DecodeImage(ops.decoders.Image):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(DecodeImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(DecodeImage, self).__call__(data, **kwargs)


class DecodeRandomResizedCrop(ops.decoders.ImageRandomCrop):
    def __init__(self,
                 *kargs,
                 device="cpu",
                 resize_x=224,
                 resize_y=224,
                 resize_short=None,
                 interp_type=types.DALIInterpType.INTERP_LINEAR,
                 **kwargs):
        super(DecodeRandomResizedCrop, self).__init__(
            *kargs, device=device, **kwargs)
        if resize_short is None:
            self.resize = ops.Resize(
                device="gpu" if device == "mixed" else "cpu",
                resize_x=resize_x,
                resize_y=resize_y,
                interp_type=interp_type)
        else:
            self.resize = ops.Resize(
                device="gpu" if device == "mixed" else "cpu",
                resize_short=resize_short,
                interp_type=interp_type)

    def __call__(self, data, **kwargs):
        data = super(DecodeRandomResizedCrop, self).__call__(data, **kwargs)
        data = self.resize(data)
        return data


class CropMirrorNormalize(ops.CropMirrorNormalize):
    def __init__(self, *kargs, device="cpu", prob=0.5, **kwargs):
        super(CropMirrorNormalize, self).__init__(
            *kargs, device=device, **kwargs)
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, data, **kwargs):
        do_mirror = self.rng()
        return super(CropMirrorNormalize, self).__call__(
            data, mirror=do_mirror, **kwargs)


class RandCropImage(ops.RandomResizedCrop):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandCropImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(RandCropImage, self).__call__(data, **kwargs)


class ResizeImage(ops.Resize):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(ResizeImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(ResizeImage, self).__call__(data, **kwargs)


class RandFlipImage(ops.Flip):
    def __init__(self, *kargs, device="cpu", prob=0.5, flip_code=1, **kwargs):
        super(RandFlipImage, self).__init__(*kargs, device=device, **kwargs)
        self.flip_code = flip_code
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, data, **kwargs):
        do_flip = self.rng()
        if self.flip_code == 1:
            return super(RandFlipImage, self).__call__(
                data, horizontal=do_flip, vertical=0, **kwargs)
        elif self.flip_code == 1:
            return super(RandFlipImage, self).__call__(
                data, horizontal=0, vertical=do_flip, **kwargs)
        else:
            return super(RandFlipImage, self).__call__(
                data, horizontal=do_flip, vertical=do_flip, **kwargs)


class Pad(ops.Pad):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(Pad, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(Pad, self).__call__(data, **kwargs)


class RandCropImageV2(ops.Crop):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandCropImageV2, self).__init__(*kargs, device=device, **kwargs)
        self.rng_x = ops.random.Uniform(range=(0.0, 1.0))
        self.rng_y = ops.random.Uniform(range=(0.0, 1.0))

    def __call__(self, data, **kwargs):
        pos_x = self.rng_x()
        pos_y = self.rng_y()
        return super(RandCropImageV2, self).__call__(
            data, crop_pos_x=pos_x, crop_pos_y=pos_y, **kwargs)


class RandomRotation(ops.Rotate):
    def __init__(self, *kargs, device="cpu", prob=0.5, angle=0, **kwargs):
        super(RandomRotation, self).__init__(*kargs, device=device, **kwargs)
        self.rng = ops.random.CoinFlip(probability=prob)
        self.rng_angle = ops.random.Uniform(range=(-angle, angle))

    def __call__(self, data, **kwargs):
        do_flip = self.rng()
        angle = self.rng_angle()
        flip_data = super(RandomRotation, self).__call__(
            data,
            angle=fn.cast(
                do_flip, dtype=types.FLOAT) * angle,
            keep_size=True,
            fill_value=0,
            **kwargs)
        return flip_data


class NormalizeImage(ops.Normalize):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(NormalizeImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(NormalizeImage, self).__call__(data, **kwargs)


INTERP_MAP = {
    "nearest": types.DALIInterpType.INTERP_NN,  # cv2.INTER_NEAREST
    "bilinear": types.DALIInterpType.INTERP_LINEAR,  # cv2.INTER_LINEAR
    "bicubic": types.DALIInterpType.INTERP_CUBIC,  # cv2.INTER_CUBIC
    "lanczos": types.DALIInterpType.
    INTERP_LANCZOS3,  # XXX use LANCZOS3 for cv2.INTER_LANCZOS4
}


def convert_cfg_to_dali(op_name: str, device: str,
                        **ops_param) -> Dict[str, Any]:
    """convert original preprocess op params into DALI-based op params

    Args:
        op_name (str): preprocess OP name

    Returns:
        Dict[str, Any]: converted params for DALI initialization
    """
    assert device in ["cpu", "gpu"
                      ], f"device({device}) must in [\"cpu\", \"gpu\"]"
    ret_dict = {}
    if op_name == "DecodeImage":
        device = "cpu" if device == "cpu" else "mixed"
        output_type = ops_param.get("output_type", types.DALIImageType.RGB)
        device_memory_padding = ops_param.get("device_memory_padding",
                                              211025920)
        host_memory_padding = ops_param.get("host_memory_padding", 140544512)
        if device is not None:
            ret_dict.update({"device": device, })
        if output_type is not None:
            ret_dict.update({"output_type": output_type, })
        if device_memory_padding is not None:
            ret_dict.update({"device_memory_padding": device_memory_padding, })
        if host_memory_padding is not None:
            ret_dict.update({"host_memory_padding": host_memory_padding, })
    elif op_name == "DecodeRandomResizedCrop":
        device = "cpu" if device == "cpu" else "mixed"
        output_type = ops_param.get("output_type", types.DALIImageType.RGB)
        device_memory_padding = ops_param.get("device_memory_padding",
                                              211025920)
        host_memory_padding = ops_param.get("host_memory_padding", 140544512)
        scale = ops_param.get("scale", [0.08, 1.0])
        ratio = ops_param.get("ratio", [3.0 / 4, 4.0 / 3])
        num_attempts = ops_param.get("num_attempts", 100)
        if device is not None:
            ret_dict.update({"device": device, })
        if output_type is not None:
            ret_dict.update({"output_type": output_type, })
        if device_memory_padding is not None:
            ret_dict.update({"device_memory_padding": device_memory_padding, })
        if host_memory_padding is not None:
            ret_dict.update({"host_memory_padding": host_memory_padding, })
        if scale is not None:
            ret_dict.update({"random_area": scale, })
        if ratio is not None:
            ret_dict.update({"random_aspect_ratio": ratio, })
        if num_attempts is not None:
            ret_dict.update({"num_attempts": num_attempts, })
    elif op_name == "CropMirrorNormalize":
        dtype = types.FLOAT16 if ops_param.get("output_fp16",
                                               False) else types.FLOAT
        output_layout = ops_param.get("output_layout", "CHW")
        crop = ops_param.get("crop", None)
        mean = ops_param.get("mean", None)
        std = ops_param.get("std", None)
        pad_output = ops_param.get("channel_num", 3) == 4
        if dtype is not None:
            ret_dict.update({"dtype": dtype, })
        if output_layout is not None:
            ret_dict.update({"output_layout": output_layout, })
        if crop is not None:
            ret_dict.update({"crop": crop, })
        if mean is not None:
            ret_dict.update({"mean": mean, })
        if std is not None:
            ret_dict.update({"std": std, })
        if pad_output is not None:
            ret_dict.update({"pad_output": pad_output, })
    elif op_name == "ResizeImage":
        size = ops_param.get("size", None)
        resize_short = ops_param.get("resize_short", None)
        interpolation = ops_param.get("interpolation", None)
        if size is not None:
            ret_dict.update({"resize_x": size[1], "resize_y": size[0]})
        if resize_short is not None:
            ret_dict.update({"resize_short": resize_short})
        if interpolation is not None:
            ret_dict.update({"interp_type": INTERP_MAP[interpolation]})
    elif op_name == "RandFlipImage":
        prob = ops_param.get("prob", 0.5)
        flip_code = ops_param.get("flip_code", 1)
        if prob is not None:
            ret_dict.update({"prob": prob})
        if flip_code is not None:
            ret_dict.update({"flip_code": flip_code})
    elif op_name == "Pad":
        padding = ops_param.get("padding", 0.5)
        fill = ops_param.get("fill", 0)
        if padding is not None:
            ret_dict.update({"shape": [224 + padding, 224 + padding, 3]})
        if fill is not None:
            ret_dict.update({"fill_value": fill})
    elif op_name == "RandCropImageV2":
        size = ops_param.get("size", None)
        if size is not None:
            ret_dict.update({"crop": size})
    elif op_name == "RandomRotation":
        prob = ops_param.get("prob", 0.5)
        degrees = ops_param.get("degrees", 90)
        interpolation = ops_param.get("interpolation", "bilinear")
        if prob is not None:
            ret_dict.update({"prob": prob})
        if degrees is not None:
            ret_dict.update({"angle": degrees})
        if interpolation is not None:
            ret_dict.update({"interp_type": INTERP_MAP[interpolation]})
    elif op_name == "NormalizeImage":
        # scale * (in - mean) / stddev + shift
        scale = ops_param.get("scale", None)
        if isinstance(scale, str):
            scale = eval(scale)
        mean = ops_param.get("mean", None)
        std = ops_param.get("std", None)
        output_fp16 = ops_param.get("output_fp16", False)
        if scale is not None:
            ret_dict.update({"scale": scale})
        if mean is not None:
            ret_dict.update({
                "mean": np.reshape(
                    np.array(
                        [v / scale for v in mean], dtype="float32"),
                    [1, 1, 3])
            })
        if std is not None:
            ret_dict.update({
                "stddev": np.reshape(
                    np.array(
                        [v / scale for v in std], dtype="float32"), [1, 1, 3])
            })
        if output_fp16 is True:
            ret_dict.update({"dtype": types.FLOAT16})
    elif op_name == "RandCropImage":
        size = ops_param.get("size")
        scale = ops_param.get("scale", [0.08, 1.0])
        ratio = ops_param.get("ratio", [3.0 / 4, 4.0 / 3])
        interpolation = ops_param.get("interpolation", "bilinear")
        if size is True:
            ret_dict.update({"size": size})
        if scale is True:
            ret_dict.update({"random_area": scale})
        if ratio is True:
            ret_dict.update({"random_aspect_ratio": ratio})
        if interpolation is True:
            ret_dict.update({"interp_type": INTERP_MAP[interpolation]})
    else:
        raise ValueError(f"Operator '{op_name}' is not implemented now.")
    if "device" not in ret_dict:
        ret_dict.update({"device": device})
    return ret_dict


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
        fuse (bool): whether to fuse transforms. Defaults to True.
    Returns:
        List[Callable]: Callable DALI operators in list.
    """
    assert isinstance(op_cfg_list, list), ('operator config should be a list')
    # build dali transforms list
    dali_ops = []
    idx = 0
    num_cfg_node = len(op_cfg_list)
    while idx < num_cfg_node:
        op_cfg = op_cfg_list[idx]
        op_name = list(op_cfg)[0]
        op_param = {} if op_cfg[op_name] is None else op_cfg[op_name]
        flag = False
        if fuse:
            if idx + 1 < num_cfg_node and (
                    op_name == "DecodeImage" and
                    list(op_cfg_list[idx + 1])[0] == "RandCropImage"):
                fused_op_name = "DecodeRandomResizedCrop"
                fused_op_param = convert_cfg_to_dali(fused_op_name, device, **{
                    ** op_param, **
                    (op_cfg_list[idx + 1][list(op_cfg_list[idx + 1])[0]])
                })
                fused_dali_op = eval(fused_op_name)(**fused_op_param)
                idx += 2
                dali_ops.append(fused_dali_op)
                flag = True
                print(
                    f"DALI Operator conversion: {fused_op_name} -> {dali_ops[-1].__class__.__name__}"
                )
            elif 0 < idx and idx + 1 < num_cfg_node and (
                    op_name == "RandFlipImage" and
                    list(op_cfg_list[idx - 1])[0] == "RandCropImage" and
                    list(op_cfg_list[idx + 1])[0] == "NormalizeImage"):
                fused_op_name = "CropMirrorNormalize"
                fused_op_param = convert_cfg_to_dali(fused_op_name, device, **{
                    ** op_param, **
                    (op_cfg_list[idx - 1][list(op_cfg_list[idx - 1])[0]]), **
                    (op_cfg_list[idx + 1][list(op_cfg_list[idx + 1])[0]])
                })
                fused_dali_op = eval(fused_op_name)(**fused_op_param)
                idx += 2
                dali_ops.append(fused_dali_op)
                flag = True
                print(
                    f"DALI Operator conversion: {fused_op_name} -> {dali_ops[-1].__class__.__name__}"
                )
        if not fuse or not flag:
            assert isinstance(op_cfg,
                              dict) and len(op_cfg) == 1, "yaml format error"
            dali_param = convert_cfg_to_dali(op_name, device, **op_param)
            dali_op = eval(op_name)(**dali_param)
            dali_ops.append(dali_op)
            idx += 1
            print(
                f"DALI Operator conversion: {op_name} -> {dali_ops[-1].__class__.__name__}"
            )
    return dali_ops


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
                 random_shuffle: bool=True):
        super(HybridPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.device = device
        self.reader = ops.readers.File(
            file_root=file_root,
            file_list=file_list,
            hard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle)
        self.transforms = ops.Compose(transform_list)
        self.cast = ops.Cast(dtype=types.DALIDataType.INT64, device=device)

    def define_graph(self):
        raw_images, labels = self.reader(name="Reader")
        images = self.transforms(raw_images)
        return [
            images, self.cast(labels.gpu() if self.device == "gpu" else labels)
        ]

    def __len__(self):
        return self.epoch_size("Reader")


class DALIImageNetIterator(DALIGenericIterator):
    def __next__(self) -> List[paddle.Tensor]:
        data_batch = super(DALIImageNetIterator,
                           self).__next__()  # List[Dict[str, Tensor], ]

        # reformat in List[Tensor1, Tensor2, ...]
        data_batch = [
            paddle.to_tensor(data_batch[0][key]) for key in self.output_map
        ]
        return data_batch


def dali_dataloader(config, mode, device, num_threads=4, seed=None):
    assert "gpu" in device, "gpu training is required for DALI"
    device_id = int(device.split(':')[1])
    config_dataloader = config[mode]
    seed = 42 if seed is None else seed
    env = os.environ
    # gpu_num = paddle.distributed.get_world_size()
    batch_size = config_dataloader["sampler"]["batch_size"]
    file_root = config_dataloader["dataset"]["image_root"]
    file_list = config_dataloader["dataset"]["cls_label_path"]

    dali_transforms = build_dali_transforms(
        shituv1_config["dataset"]["transform_ops"], device)
    if mode.lower() == "train":
        if 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env and 'FLAGS_selected_gpus' in env:
            shard_id = int(env['PADDLE_TRAINER_ID'])
            num_shards = int(env['PADDLE_TRAINERS_NUM'])
            device_id = int(env['FLAGS_selected_gpus'])
            pipe = HybridPipeline(device, batch_size, num_threads, device_id,
                                  seed + shard_id, file_root, file_list,
                                  dali_transforms, shard_id, num_shards, True)
            #  sample_per_shard = len(pipe) // num_shards
        else:
            pipe = HybridPipeline(device, batch_size, 1, device_id, seed,
                                  file_root, file_list, dali_transforms, 0, 1,
                                  False)
            #  sample_per_shard = len(pipelines[0])
        pipe.build()
        pipelines = [pipe]
        return DALIImageNetIterator(
            pipelines, ['data', 'label'], reader_name='Reader')
    else:
        sampler_name = config_dataloader["sampler"].get(
            "name", "DistributedBatchSampler")
        assert sampler_name in ["DistributedBatchSampler"], \
            f"sampler_name({sampler_name}) must in [\"DistributedBatchSampler\"]"
        # resize_shorter = transforms["ResizeImage"].get("resize_short", 256)
        # crop = transforms["CropImage"]["size"]
        if 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env and 'FLAGS_selected_gpus' in env and sampler_name == "DistributedBatchSampler":
            shard_id = int(env['PADDLE_TRAINER_ID'])
            num_shards = int(env['PADDLE_TRAINERS_NUM'])
            device_id = int(env['FLAGS_selected_gpus'])

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
            pipelines, ['data', 'label'], reader_name="Reader")


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
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "order": ""
                }
            }]
        }
    }
    dali_transforms = build_dali_transforms(
        shituv1_config["dataset"]["transform_ops"], device)
    batch_size = 3
    num_threads = 4
    seed = 42
    file_root = "/workspace/hesensen/dali_learning/DALI/docs/examples/data/images"
    pipe = HybridPipeline(
        device,
        batch_size,
        num_threads,
        device_id,
        seed,
        file_root,
        None,
        dali_transforms, )
    pipe.build()
    pipelines = [pipe]
    dali_loader = DALIImageNetIterator(
        pipelines, ['data', 'label'], reader_name='Reader')
    for iter_id, batch in enumerate(dali_loader):
        images, labels = batch
        print(images.place, images.shape)
        print(labels.place, labels.shape)
        break
