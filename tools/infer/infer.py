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


def create_predictor(args):
    def create_input():
        image = fluid.data(
            name='image', shape=[None, 3, 224, 224], dtype='float32')
        return image

    def create_model(args, model, input, class_dim=1000):
        if args.model == "GoogLeNet":
            out, _, _ = model.net(input=input, class_dim=class_dim)
        else:
            out = model.net(input=input, class_dim=class_dim)
            out = fluid.layers.softmax(out)
        return out

    model = architectures.__dict__[args.model]()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()

    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            image = create_input()
            out = create_model(args, model, image)

    infer_prog = infer_prog.clone(for_test=True)
    fluid.load(
        program=infer_prog, model_path=args.pretrained_model, executor=exe)

    return exe, infer_prog, [image.name], [out.name]


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
    exe, program, feed_names, fetch_names = create_predictor(args)

    data = preprocess(args.image_file, operators)
    data = np.expand_dims(data, axis=0)
    outputs = exe.run(program,
                      feed={feed_names[0]: data},
                      fetch_list=fetch_names,
                      return_numpy=False)
    probs = postprocess(outputs)

    for idx, prob in probs:
        print("class id: {:d}, probability: {:.4f}".format(idx, prob))


if __name__ == "__main__":
    main()
