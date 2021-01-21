# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.paddle import DALIGenericIterator

import paddle
from paddle import fluid


class HybridTrainPipe(Pipeline):
    def __init__(self,
                 file_root,
                 file_list,
                 batch_size,
                 resize_shorter,
                 crop,
                 min_area,
                 lower,
                 upper,
                 interp,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=True,
                 num_threads=4,
                 seed=42,
                 pad_output=False,
                 output_dtype=types.FLOAT):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.FileReader(
            file_root=file_root,
            file_list=file_list,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle)
        # set internal nvJPEG buffers size to handle full-sized ImageNet images
        # without additional reallocations
        device_memory_padding = 211025920
        host_memory_padding = 140544512
        self.decode = ops.ImageDecoderRandomCrop(
            device='mixed',
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            random_aspect_ratio=[lower, upper],
            random_area=[min_area, 1.0],
            num_attempts=100)
        self.res = ops.Resize(
            device='gpu', resize_x=crop, resize_y=crop, interp_type=interp)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=output_dtype,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=mean,
            std=std,
            pad_output=pad_output)
        self.coin = ops.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


class HybridValPipe(Pipeline):
    def __init__(self,
                 file_root,
                 file_list,
                 batch_size,
                 resize_shorter,
                 crop,
                 interp,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=False,
                 num_threads=4,
                 seed=42,
                 pad_output=False,
                 output_dtype=types.FLOAT):
        super(HybridValPipe, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.FileReader(
            file_root=file_root,
            file_list=file_list,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(
            device="gpu", resize_shorter=resize_shorter, interp_type=interp)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=output_dtype,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=mean,
            std=std,
            pad_output=pad_output)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


def build(config, mode='train'):
    env = os.environ
    assert config.get('use_gpu',
                      True) == True, "gpu training is required for DALI"
    assert not config.get(
        'use_aa'), "auto augment is not supported by DALI reader"
    assert float(env.get('FLAGS_fraction_of_gpu_memory_to_use', 0.92)) < 0.9, \
        "Please leave enough GPU memory for DALI workspace, e.g., by setting" \
        " `export FLAGS_fraction_of_gpu_memory_to_use=0.8`"

    dataset_config = config[mode.upper()]

    gpu_num = paddle.fluid.core.get_cuda_device_count() if (
        'PADDLE_TRAINERS_NUM') and (
            'PADDLE_TRAINER_ID'
        ) not in env else int(env.get('PADDLE_TRAINERS_NUM', 0))

    batch_size = dataset_config.batch_size
    assert batch_size % gpu_num == 0, \
        "batch size must be multiple of number of devices"
    batch_size = batch_size // gpu_num

    file_root = dataset_config.data_dir
    file_list = dataset_config.file_list

    interp = 1  # settings.interpolation or 1  # default to linear
    interp_map = {
        0: types.INTERP_NN,  # cv2.INTER_NEAREST
        1: types.INTERP_LINEAR,  # cv2.INTER_LINEAR
        2: types.INTERP_CUBIC,  # cv2.INTER_CUBIC
        4: types.INTERP_LANCZOS3,  # XXX use LANCZOS3 for cv2.INTER_LANCZOS4
    }

    output_dtype = (types.FLOAT16 if 'AMP' in config and
                    config.AMP.get("use_pure_fp16", False) 
                    else types.FLOAT)
    
    assert interp in interp_map, "interpolation method not supported by DALI"
    interp = interp_map[interp]
    pad_output = False
    image_shape = config.get("image_shape", None)
    if image_shape and image_shape[0] == 4:
        pad_output = True

    transforms = {
        k: v
        for d in dataset_config["transforms"] for k, v in d.items()
    }

    scale = transforms["NormalizeImage"].get("scale", 1.0 / 255)
    if isinstance(scale, str):
        scale = eval(scale)
    mean = transforms["NormalizeImage"].get("mean", [0.485, 0.456, 0.406])
    std = transforms["NormalizeImage"].get("std", [0.229, 0.224, 0.225])
    mean = [v / scale for v in mean]
    std = [v / scale for v in std]

    if mode == "train":
        resize_shorter = 256
        crop = transforms["RandCropImage"]["size"]
        scale = transforms["RandCropImage"].get("scale", [0.08, 1.])
        ratio = transforms["RandCropImage"].get("ratio", [3.0 / 4, 4.0 / 3])
        min_area = scale[0]
        lower = ratio[0]
        upper = ratio[1]

        if 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env:
            shard_id = int(env['PADDLE_TRAINER_ID'])
            num_shards = int(env['PADDLE_TRAINERS_NUM'])
            device_id = int(env['FLAGS_selected_gpus'])
            pipe = HybridTrainPipe(
                file_root,
                file_list,
                batch_size,
                resize_shorter,
                crop,
                min_area,
                lower,
                upper,
                interp,
                mean,
                std,
                device_id,
                shard_id,
                num_shards,
                seed=42 + shard_id,
                pad_output=pad_output,
                output_dtype=output_dtype)
            pipe.build()
            pipelines = [pipe]
            sample_per_shard = len(pipe) // num_shards
        else:
            pipelines = []
            places = fluid.framework.cuda_places()
            num_shards = len(places)
            for idx, p in enumerate(places):
                place = fluid.core.Place()
                place.set_place(p)
                device_id = place.gpu_device_id()
                pipe = HybridTrainPipe(
                    file_root,
                    file_list,
                    batch_size,
                    resize_shorter,
                    crop,
                    min_area,
                    lower,
                    upper,
                    interp,
                    mean,
                    std,
                    device_id,
                    idx,
                    num_shards,
                    seed=42 + idx,
                pad_output=pad_output,
                output_dtype=output_dtype)
                pipe.build()
                pipelines.append(pipe)
            sample_per_shard = len(pipelines[0])
        return DALIGenericIterator(
            pipelines, ['feed_image', 'feed_label'], size=sample_per_shard)
    else:
        resize_shorter = transforms["ResizeImage"].get("resize_short", 256)
        crop = transforms["CropImage"]["size"]

        p = fluid.framework.cuda_places()[0]
        place = fluid.core.Place()
        place.set_place(p)
        device_id = place.gpu_device_id()
        pipe = HybridValPipe(
            file_root,
            file_list,
            batch_size,
            resize_shorter,
            crop,
            interp,
            mean,
            std,
            device_id=device_id,
            pad_output=pad_output,
            output_dtype=output_dtype)
        pipe.build()
        return DALIGenericIterator(
            pipe, ['feed_image', 'feed_label'],
            size=len(pipe),
            dynamic_shape=True,
            fill_last_batch=True,
            last_batch_padded=True)


def train(config):
    return build(config, 'train')


def val(config):
    return build(config, 'valid')


def _to_Tensor(lod_tensor, dtype):
    data_tensor = fluid.layers.create_tensor(dtype=dtype)
    data = np.array(lod_tensor).astype(dtype)
    fluid.layers.assign(data, data_tensor)
    return data_tensor


def normalize(feeds, config):
    image, label = feeds['image'], feeds['label']
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image = fluid.layers.cast(image, 'float32')
    costant = fluid.layers.fill_constant(
        shape=[1], value=255.0, dtype='float32')
    image = fluid.layers.elementwise_div(image, costant)

    mean = fluid.layers.create_tensor(dtype="float32")
    fluid.layers.assign(input=img_mean.astype("float32"), output=mean)
    std = fluid.layers.create_tensor(dtype="float32")
    fluid.layers.assign(input=img_std.astype("float32"), output=std)

    image = fluid.layers.elementwise_sub(image, mean)
    image = fluid.layers.elementwise_div(image, std)

    image.stop_gradient = True
    feeds['image'] = image

    return feeds


def mix(feeds, config, is_train=True):
    env = os.environ
    gpu_num = paddle.fluid.core.get_cuda_device_count() if (
        'PADDLE_TRAINERS_NUM') and (
            'PADDLE_TRAINER_ID'
        ) not in env else int(env.get('PADDLE_TRAINERS_NUM', 0))

    batch_size = config.TRAIN.batch_size // gpu_num

    images = feeds['image']
    label = feeds['label']
    # TODO: hard code here, should be fixed!
    alpha = 0.2
    idx = _to_Tensor(np.random.permutation(batch_size), 'int32')
    lam = np.random.beta(alpha, alpha)

    images = lam * images + (1 - lam) * paddle.fluid.layers.gather(images, idx)

    feed = {
        'image': images,
        'feed_y_a': label,
        'feed_y_b': paddle.fluid.layers.gather(label, idx),
        'feed_lam': _to_Tensor([lam] * batch_size, 'float32')
    }

    return feed if is_train else feeds
