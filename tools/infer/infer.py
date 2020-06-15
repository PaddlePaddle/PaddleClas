# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import utils
import argparse
import numpy as np

import paddle.fluid as fluid
from ppcls.modeling import architectures

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_file", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-p", "--pretrained_model", type=str)
    parser.add_argument("--use_gpu", type=str2bool, default=True)

    return parser.parse_args()

def create_operators():
    size = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    decode_op = utils.DecodeImage()
    resize_op = utils.ResizeImage(resize_short=256)
    crop_op = utils.CropImage(size=(size, size))
    normalize_op = utils.NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = utils.ToTensor()

    return [decode_op, resize_op, crop_op, normalize_op, totensor_op]


def preprocess(fname, ops):
    data = open(fname, 'rb').read()
    for op in ops:
        data = op(data)

    return data


def postprocess(outputs, topk=5):
    output = outputs[0]
    prob = np.array(output).flatten()
    index = prob.argsort(axis=0)[-topk:][::-1].astype('int32')
    return zip(index, prob[index])


def main():
    args = parse_args()
    operators = create_operators()
    # assign the place
    gpu_id = fluid.dygraph.parallel.Env().dev_id
    place = fluid.CUDAPlace(gpu_id)
    
    pre_weights_dict = fluid.load_program_state(args.pretrained_model)
    with fluid.dygraph.guard(place):
        net = architectures.__dict__[args.model]()
        data = preprocess(args.image_file, operators)
        data = np.expand_dims(data, axis=0)
        data = fluid.dygraph.to_variable(data)
        dy_weights_dict = net.state_dict()
        pre_weights_dict_new = {}
        for key in dy_weights_dict:
            weights_name = dy_weights_dict[key].name
            pre_weights_dict_new[key] = pre_weights_dict[weights_name]
        net.set_dict(pre_weights_dict_new)
        net.eval()
        outputs = net(data)
        outputs = fluid.layers.softmax(outputs)
        outputs = outputs.numpy()
        
    probs = postprocess(outputs)
    rank = 1
    for idx, prob in probs:
        print("top{:d}, class id: {:d}, probability: {:.4f}".format(
            rank, idx, prob))
        rank += 1

if __name__ == "__main__":
    main()
